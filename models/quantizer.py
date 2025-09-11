import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(
    logits,
    temperature = 1.,
    stochastic = False,
    dim = -1,
    training = True
):

    if training and stochastic and temperature > 0:
        sampling_logits = (logits / temperature) + gumbel_noise(logits)
    else:
        sampling_logits = logits

    ind = sampling_logits.argmax(dim = dim)

    return ind
class Quantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(Quantizer, self).__init__()

        self.e_dim = e_dim
        self.n_e = n_e
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vectort that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        :param z (B, seq_len, channel):
        :return z_q:
        """
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)

        # B x V
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q - z.detach())**2) + self.beta * \
               torch.mean((z_q.detach() - z)**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        min_encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean*torch.log(e_mean + 1e-10)))
        return loss, z_q, min_encoding_indices, perplexity

    def map2index(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vectort that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        :param z (B, seq_len, channel):
        :return z_q:
        """
        assert z.shape[-1] == self.e_dim
        #print(z.shape)
        z_flattened = z.contiguous().view(-1, self.e_dim)
        #print(z_flattened.shape)

        # B x V
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        return min_encoding_indices.reshape(z.shape[0], -1)

    def get_codebook_entry(self, indices):
        """

        :param indices(B, seq_len):
        :return z_q(B, seq_len, e_dim):
        """
        index_flattened = indices.view(-1)
        z_q = self.embedding(index_flattened)
        z_q = z_q.view(indices.shape + (self.e_dim, )).contiguous()
        return z_q


class EmbeddingEMA(nn.Module):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5):
        super(EmbeddingEMA, self).__init__()
        self.decay = decay
        self.eps = eps
        weight = torch.randn(num_tokens, codebook_dim)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad=False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad=False)
        self.update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_emb_avg):
        self.embed_avg.data.mul_(self.decay).add(new_emb_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
            (self.cluster_size + self.eps) / (n + num_tokens*self.eps) * n
        )
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized)


class EMAVectorQuantizer(nn.Module):
    def __init__(self, n_embed, embedding_dim, beta, decay=0.99, eps=1e-5):
        super(EMAVectorQuantizer, self).__init__()
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.beta = beta
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps)

    def forward(self, z):
        z_flattened = z.view(-1, self.codebook_dim)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        min_encodings = F.one_hot(min_encoding_indices, self.num_tokens).type(z.dtype)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        if self.training and self.embedding.update:
            encoding_sum = min_encodings.sum(0)
            embed_sum = min_encodings.transpose(0, 1)@z_flattened

            self.embedding.cluster_size_ema_update(encoding_sum)
            self.embedding.embed_avg_ema_update(embed_sum)
            self.embedding.weight_update(self.num_tokens)

        loss = self.beta * F.mse_loss(z_q.detach(), z)

        z_q = z + (z_q - z).detach()
        return loss, z_q, min_encoding_indices, perplexity


# class GumbelQuantizer(nn.Module):
#     def __init__(self, num_hiddens, embedding_dim, n_embed, straight_through=True,
#                  kl_weight=5e-4, temp_init=1.0):
#         super(GumbelQuantizer, self).__init__()
#
#         self.embedding_dim = embedding_dim
#         self.n_embed = n_embed
#
#         self.straight_through = straight_through
#         self.temperature = temp_init
#         self.kl_weight = kl_weight
#
#         self.proj = nn.Linear(num_hiddens, n_embed)
#         self.embed = nn.Embedding(n_embed, embedding_dim)
class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, args):
        super(QuantizeEMAReset, self).__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = 0.99  ##TO_DO
        self.reset_codebook()

    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim, requires_grad=False).cuda())

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    def quantize(self, x, sample_codebook_temp=0.):
        # N X C -> C X N
        k_w = self.codebook.t()
        # x: NT X C
        # NT X N
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - \
                   2 * torch.matmul(x, k_w) + \
                   torch.sum(k_w ** 2, dim=0, keepdim=True)  # (N * L, b)

        # code_idx = torch.argmin(distance, dim=-1)

        code_idx = gumbel_sample(-distance, dim = -1, temperature = sample_codebook_temp, stochastic=True, training = self.training)

        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x
    
    def get_codebook_entry(self, indices):
        return self.dequantize(indices).permute(0, 2, 1)

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device) # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x) # nb_code, c
        code_count = code_onehot.sum(dim=-1) # nb_code

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)
        self.codebook = usage * code_update + (1-usage) * code_rand


        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        # x = x.permute(0, 2, 1).contiguous()
        # x = x.view(-1, x.shape[-1])
        x = rearrange(x, 'n c t -> (n t) c')
        return x

    def forward(self, x, return_idx=False, temperature=0.):
        N, width, T = x.shape

        x = self.preprocess(x)
        if self.training and not self.init:
            self.init_codebook(x)

        code_idx = self.quantize(x, temperature)
        x_d = self.dequantize(code_idx)

        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)
        # print('before x', x)
        # print('before x_d', x_d)
        commit_loss = F.mse_loss(x, x_d.detach()) # It's right. the t2m-gpt paper is wrong on embed loss and commitment loss.
        
        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        code_idx = code_idx.view(N, T).contiguous()
        # print(code_idx[0])
        if return_idx:
            return x_d, code_idx, commit_loss, perplexity
        return x_d, commit_loss, perplexity

class QuantizeEMAReset2D(nn.Module):
    """Quantize EMA supporting 2D Joint-Temporal Quantization"""
    def __init__(self, nb_code, code_dim, mu=0.99, args=None):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = mu

        self.register_buffer('codebook', torch.randn(nb_code, code_dim))  # (nb_code, code_dim)
        self.register_buffer('code_sum', torch.zeros(nb_code, code_dim))
        self.register_buffer('code_count', torch.ones(nb_code))

    def preprocess(self, x):
        """
        Flatten J and T for quantization.
        Input: (B, C, J, T)
        Output: (B * J * T, C)
        """
        return rearrange(x, 'b c j t -> (b j t) c')

    def postprocess(self, x, B, J, T):
        """
        Reshape quantized output back to original dimensions.
        Input: (B * J * T, C)
        Output: (B, C, J, T)
        """
        return rearrange(x, '(b j t) c -> b c j t', b=B, j=J, t=T)

    def forward(self, x, return_idx=False, temperature=0.):
        """
        Args:
            x: (B, C, J, T)
        Returns:
            x_quantized: (B, C, J, T)
            code_idx: (B, J, T)
            commit_loss: scalar
            perplexity: scalar
        """
        B, C, J, T = x.shape

        # Flatten (B, C, J, T) -> (B * J * T, C)
        x_flattened = self.preprocess(x)

        # Compute distances to codebook entries
        distances = (
            torch.sum(x_flattened ** 2, dim=-1, keepdim=True)  # (N, 1)
            - 2 * torch.matmul(x_flattened, self.codebook.t())  # (N, nb_code)
            + torch.sum(self.codebook ** 2, dim=-1).unsqueeze(0)  # (1, nb_code)
        )

        # Find nearest codebook entry
        code_idx = torch.argmin(distances, dim=-1)  # (N,)

        # Quantize
        x_quantized = F.embedding(code_idx, self.codebook)  # (N, C)

        # Compute commitment loss
        commit_loss = F.mse_loss(x_flattened, x_quantized.detach())

        # Update codebook with EMA
        if self.training:
            self._update_codebook(x_flattened, code_idx)

        # Reshape quantized output back to (B, C, J, T)
        x_quantized = self.postprocess(x_quantized, B, J, T)

        # Reshape code indices back to (B, J, T)
        code_idx = rearrange(code_idx, '(b j t) -> b j t', b=B, j=J, t=T)

        if return_idx:
            return x_quantized, code_idx, commit_loss, self._compute_perplexity(code_idx)

        return x_quantized, commit_loss

    @torch.no_grad()
    def _update_codebook(self, x, code_idx):
        """
        Update codebook with EMA.
        Args:
            x: (N, C)
            code_idx: (N,)
        """
        one_hot = F.one_hot(code_idx, self.nb_code).float()  # (N, nb_code)
        code_sum = torch.matmul(one_hot.t(), x)  # (nb_code, C)
        code_count = one_hot.sum(dim=0)  # (nb_code,)

        self.code_sum = self.mu * self.code_sum + (1 - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1 - self.mu) * code_count

        usage = (self.code_count >= 1.0).float().unsqueeze(-1)
        self.codebook = usage * (self.code_sum / self.code_count.unsqueeze(-1)) + (1 - usage) * self.codebook

    @torch.no_grad()
    def _compute_perplexity(self, code_idx):
        """
        Compute perplexity.
        Args:
            code_idx: (N,)
        Returns:
            perplexity: scalar
        """
        one_hot = F.one_hot(code_idx, self.nb_code).float()
        code_count = one_hot.sum(dim=0)  # (nb_code,)
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
