import random
from math import ceil
from functools import partial
from itertools import zip_longest
from random import randrange

import torch
from torch import nn
import torch.nn.functional as F
# from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize
from .quantizer import QuantizeEMAReset, QuantizeEMAReset2D

from einops import rearrange, repeat, pack, unpack

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

# main class

class ResidualVQ(nn.Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """
    def __init__(
        self,
        num_quantizers,
        shared_codebook=False,
        quantize_dropout_prob=0.5,
        quantize_dropout_cutoff_index=0,
        **kwargs
    ):
        super().__init__()

        self.num_quantizers = num_quantizers

        # self.layers = nn.ModuleList([VectorQuantize(accept_image_fmap = accept_image_fmap, **kwargs) for _ in range(num_quantizers)])
        if shared_codebook:
            layer = QuantizeEMAReset(**kwargs)
            self.layers = nn.ModuleList([layer for _ in range(num_quantizers)])
        else:
            self.layers = nn.ModuleList([QuantizeEMAReset(**kwargs) for _ in range(num_quantizers)])
        # self.layers = nn.ModuleList([QuantizeEMA(**kwargs) for _ in range(num_quantizers)])

        # self.quantize_dropout = quantize_dropout and num_quantizers > 1

        assert quantize_dropout_cutoff_index >= 0 and quantize_dropout_prob >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_prob = quantize_dropout_prob

            
    @property
    def codebooks(self):
        codebooks = [layer.codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim = 0)
        return codebooks # 'q c d'
    
    def get_codes_from_indices(self, indices): #indices shape 'b n q' # dequantize

        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # because of quantize dropout, one can pass in indices that are coarse
        # and the network should be able to reconstruct

        if quantize_dim < self.num_quantizers:
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value = -1)

        # get ready for gathering

        codebooks = repeat(self.codebooks, 'q c d -> q b c d', b = batch)
        gather_indices = repeat(indices, 'b n q -> q b n d', d = codebooks.shape[-1])

        # take care of quantizer dropout

        mask = gather_indices == -1.
        gather_indices = gather_indices.masked_fill(mask, 0) # have it fetch a dummy code to be masked out later

        # print(gather_indices.max(), gather_indices.min())
        all_codes = codebooks.gather(2, gather_indices) # gather all codes

        # mask out any codes that were dropout-ed

        all_codes = all_codes.masked_fill(mask, 0.)

        return all_codes # 'q b n d'

    def get_codebook_entry(self, indices): #indices shape 'b n q'
        all_codes = self.get_codes_from_indices(indices) #'q b n d'
        latent = torch.sum(all_codes, dim=0) #'b n d'
        latent = latent.permute(0, 2, 1)
        return latent

    def forward(self, x, return_all_codes=False, sample_codebook_temp=None, force_dropout_index=-1):
        num_quant = self.num_quantizers
        quantized_out = 0.
        residual = x
        all_losses = []
        all_indices = []
        all_perplexity = []
        device = x.device
        
        should_quantize_dropout = self.training and random.random() < self.quantize_dropout_prob
        start_drop_quantize_index = num_quant

        # if should_quantize_dropout:
        #     start_drop_quantize_index = randrange(self.quantize_dropout_cutoff_index, num_quant)
        #     null_indices_shape = [x.shape[0], x.shape[-1]]
        #     null_indices = torch.full(null_indices_shape, -1, device=device, dtype=torch.long)

        # if force_dropout_index >= 0:
        #     should_quantize_dropout = True
        #     start_drop_quantize_index = force_dropout_index
        #     null_indices_shape = [x.shape[0], x.shape[-1]]
        #     null_indices = torch.full(null_indices_shape, -1, device=device, dtype=torch.long)
        
        for quantizer_index, layer in enumerate(self.layers):
            # if should_quantize_dropout and quantizer_index > start_drop_quantize_index:
            #     all_indices.append(null_indices)
            #     continue
            quantized, *rest = layer(residual, return_idx=True, temperature=sample_codebook_temp)
            residual -= quantized.detach()
            quantized_out += quantized
            embed_indices, loss, perplexity = rest

            if torch.isnan(quantized).any() or torch.isinf(quantized).any():
                print(f"NaN or Inf detected in quantizer output at layer {quantizer_index}")

            all_indices.append(embed_indices)
            all_losses.append(loss)
            all_perplexity.append(perplexity)

        all_indices = torch.stack(all_indices, dim=-1)
        all_losses = sum(all_losses) / len(all_losses)
        all_perplexity = sum(all_perplexity) / len(all_perplexity)

        if torch.isnan(all_losses).any() or torch.isinf(all_losses).any():
            print("NaN or Inf detected in accumulated quantizer losses")

        ret = (quantized_out, all_indices, all_losses, all_perplexity)

        if return_all_codes:
            all_codes = self.get_codes_from_indices(all_indices)
            ret = (*ret, all_codes)

        return ret
    
    def quantize(self, x, return_latent=False):
        all_indices = []
        quantized_out = 0.
        residual = x
        all_codes = []
        for quantizer_index, layer in enumerate(self.layers):

            quantized, *rest = layer(residual, return_idx=True) #single quantizer

            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            embed_indices, loss, perplexity = rest
            all_indices.append(embed_indices)
            # print(quantizer_index, embed_indices[0])
            # print(quantizer_index, quantized[0])
            # break
            all_codes.append(quantized)

        code_idx = torch.stack(all_indices, dim=-1)
        all_codes = torch.stack(all_codes, dim=0)
        if return_latent:
            return code_idx, all_codes
        return code_idx
    
class ResidualVQ_2D(nn.Module):
    """Residual Vector Quantization supporting 2D Joint-Temporal Quantization"""
    def __init__(
        self,
        num_quantizers,
        shared_codebook=False,
        quantize_dropout_prob=0.5,
        quantize_dropout_cutoff_index=0,
        **kwargs
    ):
        super().__init__()

        self.num_quantizers = num_quantizers

        if shared_codebook:
            layer = QuantizeEMAReset2D(**kwargs)
            self.layers = nn.ModuleList([layer for _ in range(num_quantizers)])
        else:
            self.layers = nn.ModuleList([QuantizeEMAReset2D(**kwargs) for _ in range(num_quantizers)])

        self.quantize_dropout_prob = quantize_dropout_prob
        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index

    def forward(self, x, return_all_codes=False, sample_codebook_temp=None):
        """
        Args:
            x: (B, C, J, T)
        Returns:
            quantized_out: (B, C, J, T)
            all_indices: (B, J, T, num_quantizers)
            all_losses: scalar
            all_perplexity: scalar
        """
        quantized_out = 0.0
        residual = x
        all_indices = []
        all_losses = []
        all_perplexity = []

        for layer in self.layers:
            quantized, indices, loss, perplexity = layer(residual, return_idx=True, temperature=sample_codebook_temp)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
            all_losses.append(loss)
            all_perplexity.append(perplexity)

        all_indices = torch.stack(all_indices, dim=-1)  # (B, J, T, num_quantizers)
        all_losses = sum(all_losses) / len(all_losses)
        all_perplexity = sum(all_perplexity) / len(all_perplexity)

        ret = (quantized_out, all_indices, all_losses, all_perplexity)

        if return_all_codes:
            all_codes = [layer.get_codebook_entry(indices) for layer, indices in zip(self.layers, all_indices)]
            ret = (*ret, all_codes)

        return ret
class ResidualVQ_hrmg(nn.Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """
    def __init__(
        self,
        num_quantizers,
        shared_codebook=False,
        quantize_dropout_prob=0.2,
        quantize_dropout_cutoff_index=0,
        **kwargs
    ):
        super().__init__()

        self.num_quantizers = num_quantizers

        # self.layers = nn.ModuleList([VectorQuantize(accept_image_fmap = accept_image_fmap, **kwargs) for _ in range(num_quantizers)])

        self.layers_scale1 = nn.ModuleList([QuantizeEMAReset(**kwargs) for _ in range(num_quantizers)])
        self.layers_scale2 = nn.ModuleList([QuantizeEMAReset(**kwargs) for _ in range(num_quantizers)])
        self.layers_scale3 = nn.ModuleList([QuantizeEMAReset(**kwargs) for _ in range(num_quantizers)])
        self.layers_scale4 = nn.ModuleList([QuantizeEMAReset(**kwargs) for _ in range(num_quantizers)])
        self.layers_scale_full = nn.ModuleList([QuantizeEMAReset(**kwargs) for _ in range(num_quantizers)])
        # self.layers = nn.ModuleList([QuantizeEMA(**kwargs) for _ in range(num_quantizers)])

        # self.quantize_dropout = quantize_dropout and num_quantizers > 1

        assert quantize_dropout_cutoff_index >= 0 and quantize_dropout_prob >= 0

        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_prob = quantize_dropout_prob

            
    @property
    def codebooks(self):
        codebooks = [layer.codebook for layer in self.layers]
        codebooks = torch.stack(codebooks, dim = 0)
        return codebooks # 'q c d'
    
    def get_codes_from_indices(self, indices): #indices shape 'b n q' # dequantize

        batch, quantize_dim = indices.shape[0], indices.shape[-1]

        # because of quantize dropout, one can pass in indices that are coarse
        # and the network should be able to reconstruct

        if quantize_dim < self.num_quantizers:
            indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value = -1)

        # get ready for gathering

        codebooks = repeat(self.codebooks, 'q c d -> q b c d', b = batch)
        gather_indices = repeat(indices, 'b n q -> q b n d', d = codebooks.shape[-1])

        # take care of quantizer dropout

        mask = gather_indices == -1.
        gather_indices = gather_indices.masked_fill(mask, 0) # have it fetch a dummy code to be masked out later

        # print(gather_indices.max(), gather_indices.min())
        all_codes = codebooks.gather(2, gather_indices) # gather all codes

        # mask out any codes that were dropout-ed

        all_codes = all_codes.masked_fill(mask, 0.)

        return all_codes # 'q b n d'

    def get_codebook_entry(self, indices): #indices shape 'b n q'
        all_codes = self.get_codes_from_indices(indices) #'q b n d'
        latent = torch.sum(all_codes, dim=0) #'b n d'
        latent = latent.permute(0, 2, 1)
        return latent

    def forward(self, x, return_all_codes = False, sample_codebook_temp = None, force_dropout_index=-1):
        # debug check
        # print(self.codebooks[:,0,0].detach().cpu().numpy())
        num_quant, quant_dropout_prob, device = self.num_quantizers, self.quantize_dropout_prob, x.device

        quantized_out = 0.
        residual = x

        all_losses = []
        all_indices = []
        all_perplexity = []


        should_quantize_dropout = self.training and random.random() < self.quantize_dropout_prob

        start_drop_quantize_index = num_quant
        # To ensure the first-k layers learn things as much as possible, we randomly dropout the last q - k layers
        if should_quantize_dropout:
            start_drop_quantize_index = randrange(self.quantize_dropout_cutoff_index, num_quant) # keep quant layers <= quantize_dropout_cutoff_index, TODO vary in batch
            null_indices_shape = [x.shape[0], x.shape[-1]] # 'b*n'
            null_indices = torch.full(null_indices_shape, -1., device = device, dtype = torch.long)
            # null_loss = 0.

        if force_dropout_index >= 0:
            should_quantize_dropout = True
            start_drop_quantize_index = force_dropout_index
            null_indices_shape = [x.shape[0], x.shape[-1]]  # 'b*n'
            null_indices = torch.full(null_indices_shape, -1., device=device, dtype=torch.long)
        # print(force_dropout_index)
        # go through the layers
        
        all_ret = {
            "scale1": {},
            "scale2": {},
            "scale3": {},
            "scale4": {},
            "scale_full": {}
            }
        # all_ret = {
        #     "scale4": {}
        #     }
        # for i, (scale_name, layers) in enumerate(zip(all_ret.keys(), [self.layers_scale4]), 1):
        for i, (scale_name, layers) in enumerate(zip(all_ret.keys(), [self.layers_scale1, self.layers_scale2, self.layers_scale3, self.layers_scale4, self.layers_scale_full]), 1):
            all_indices = []
            all_losses = []
            all_perplexity = []

            for quantizer_index, layer in enumerate(layers):
                if should_quantize_dropout and quantizer_index > start_drop_quantize_index:
                    all_indices.append(null_indices)
                    continue
                quantized, *rest = layer(residual, return_idx=True, temperature=sample_codebook_temp)
                residual -= quantized.detach()
                quantized_out += quantized
                embed_indices, loss, perplexity = rest
                all_indices.append(embed_indices)
                all_losses.append(loss)
                all_perplexity.append(perplexity)
                
            # stack all losses and indices for the current scale
            all_indices = torch.stack(all_indices, dim=-1)
            all_losses = sum(all_losses) / len(all_losses)
            all_perplexity = sum(all_perplexity) / len(all_perplexity)
            
            # store the results for the current scale in the dictionary
            all_ret[scale_name]["quantized_out"] = quantized_out
            all_ret[scale_name]["all_indices"] = all_indices
            all_ret[scale_name]["all_losses"] = all_losses
            all_ret[scale_name]["all_perplexity"] = all_perplexity

        return all_ret

    
    def quantize(self, x, return_latent=False):
        all_indices = []
        quantized_out = 0.
        residual = x
        all_codes = []
        for quantizer_index, layer in enumerate(self.layers):

            quantized, *rest = layer(residual, return_idx=True) #single quantizer

            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            embed_indices, loss, perplexity = rest
            all_indices.append(embed_indices)
            # print(quantizer_index, embed_indices[0])
            # print(quantizer_index, quantized[0])
            # break
            all_codes.append(quantized)

        code_idx = torch.stack(all_indices, dim=-1)
        all_codes = torch.stack(all_codes, dim=0)
        if return_latent:
            return code_idx, all_codes
        return code_idx