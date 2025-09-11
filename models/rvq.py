import random

import torch.nn as nn
from .encdec import Encoder, Decoder, Encoder2d, Decoder2d, Encoder_s, Decoder_s
from .residual_vq import ResidualVQ, ResidualVQ_hrmg, ResidualVQ_2D
import torch
# from .causal_decoder import Causal_Decoder
from einops import rearrange


class RVQVAE_2D(nn.Module):

    def __init__(self,
                 args,
                 nb_code=256,
                 code_dim=256,
                 output_emb_width=256,
                 down_t_1d=2,
                 stride_t_1d=2,
                 down_t_2d=3,
                 stride_t_2d=2,
                 width=256,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):

        super().__init__()
        self.code_dim1d = code_dim
        self.num_code1d = nb_code
        input_width = args.vae_test_dim
        self.encoder1d = Encoder(input_width,
                                 self.code_dim1d,
                                 down_t_1d,
                                 stride_t_1d,
                                 width,
                                 depth,
                                 dilation_growth_rate,
                                 activation=activation,
                                 norm=norm)
        self.decoder1d = Decoder(input_width,
                                 self.code_dim1d,
                                 down_t_1d,
                                 stride_t_1d,
                                 width,
                                 depth,
                                 dilation_growth_rate,
                                 activation=activation,
                                 norm=norm)

        rvqvae_config = {
            'num_quantizers': 6,
            'shared_codebook': False,
            'quantize_dropout_prob': 0.2,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim': code_dim,
            'args': args,
        }
        self.quantizer1d = ResidualVQ(**rvqvae_config)
        self.J0 = input_width // 6
        self.J = (self.J0 + 7) // 8
        self.JD0 = 6
        self.encode_dim2d = self.JD0
        self.num_code2d = nb_code  ###########
        self.code_dim2d = code_dim  ###########
        self.encoder2d = Encoder2d(self.encode_dim2d,
                                   self.code_dim2d,
                                   down_t_2d,
                                   stride_t_2d,
                                   width,
                                   depth,
                                   dilation_growth_rate,
                                   activation=activation,
                                   norm=norm)
        self.linear_merge = nn.Linear(self.code_dim2d * self.J,
                                      self.code_dim2d * self.J)
        self.decoder2d = Decoder2d(self.encode_dim2d,
                                   self.code_dim2d,
                                   down_t_2d,
                                   stride_t_2d,
                                   width,
                                   depth,
                                   dilation_growth_rate,
                                   activation=activation,
                                   norm=norm)
        self.linear_out = nn.Linear(input_width * 2, input_width)
        joints_rvqvae_config = {
            'num_quantizers': 6,
            'shared_codebook': False,
            'quantize_dropout_prob': 0.2,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': self.num_code2d,
            'code_dim': self.code_dim2d,
            'args': args,
        }
        self.quantizer2d = ResidualVQ(**joints_rvqvae_config)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x1d, x2d):
        x_in = self.preprocess(x1d)
        # Encode
        x_encoder1d = self.encoder1d(x_in)
        ## quantization
        x_quantized1d, code_idx1d, commit_loss1d, perplexity1d = self.quantizer1d(
            x_encoder1d, sample_codebook_temp=0.5)
        ## decoder
        x_out1d = self.decoder1d(x_quantized1d)

        B, T0, _, _ = x2d.shape
        if self.J0 == 30:
            x2d = torch.nn.functional.pad(x2d,
                                          (0, 0, 1, 1))  # for downsample X 2
        elif self.J0 == 13:
            x2d = torch.nn.functional.pad(x2d,
                                          (0, 0, 1, 2))  # for downsample X 2
        elif self.J0 == 11:
            x2d = torch.nn.functional.pad(x2d, (0, 0, 2, 3))
        x2d_encode = self.encoder2d(rearrange(x2d, 'b t j d -> b d t j'))
        B, D, T, _ = x2d_encode.shape

        x2d_quantized, code_idx2d, commit_loss2d, perplexity2d = self.quantizer2d(
            rearrange(x2d_encode, 'b d t j -> (b j) d t'),
            sample_codebook_temp=0.5)
        x2d_quantized = x2d_quantized.reshape(B, self.J * D, T)
        x2d_quantized = self.linear_merge(
            x2d_quantized.permute(0, 2, 1).reshape(B * T, self.J * D)).reshape(
                B, T, self.J * D).permute(0, 2, 1)
        x2d_out = self.decoder2d(
            rearrange(x2d_quantized.reshape(B, self.J, D, T),
                      'b j d t -> b d t j'))

        T0 = x2d_out.shape[1]
        if self.J0 == 30:
            x2d_out = x2d_out.reshape(B, T0, self.J0 + 2,
                                      self.JD0)[:, :,
                                                1:-1, :].reshape(B, T0, -1)
        elif self.J0 == 13:
            x2d_out = x2d_out.reshape(B, T0, self.J0 + 3,
                                      self.JD0)[:, :,
                                                1:-2, :].reshape(B, T0, -1)
        elif self.J0 == 11:
            x2d_out = x2d_out.reshape(B, T0, self.J0 + 5,
                                      self.JD0)[:, :,
                                                2:-3, :].reshape(B, T0, -1)
        x_out2d = self.linear_out(torch.concat([x_out1d, x2d_out], dim=2))
        output = {
            'rec_pose_1d': x_out1d,
            'rec_pose_2d': x_out2d,
            'embedding_loss_1d': commit_loss1d,
            'embedding_loss_2d': commit_loss2d,
        }
        return output

    def map2zp(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        return x_encoder

    def decode(self, x1d=None, x2d=None):
        x_out1d = None
        if x1d is not None:
            x_d = self.quantizer1d.get_codes_from_indices(x1d)
            x1d = x_d.sum(dim=0).permute(0, 2, 1)
            x_out1d = self.decoder1d(x1d)

        x2d_out = None
        if x2d is not None:
            B, T, _, _ = x2d.shape
            x_d_2d = self.quantizer2d.get_codes_from_indices(
                rearrange(x2d, 'b t j n -> (b j) t n'))
            D = x_d_2d.shape[-1]
            x2d = rearrange(
                x_d_2d.sum(dim=0).reshape(B, self.J, T, D),
                'b j t d -> b (j d) t')
            x2d_quantized = self.linear_merge(
                x2d.permute(0, 2, 1).reshape(B * T, self.J * D)).reshape(
                    B, T, self.J * D).permute(0, 2, 1)
            x2d_out = self.decoder2d(
                rearrange(x2d_quantized.reshape(B, self.J, D, T),
                          'b j d t -> b d t j'))
            T0 = x2d_out.shape[1]
            if self.J0 == 30:
                x2d_out = x2d_out.reshape(B, T0, self.J0 + 2,
                                          self.JD0)[:, :, 1:-1, :].reshape(
                                              B, T0, -1)
            elif self.J0 == 13:
                x2d_out = x2d_out.reshape(B, T0, self.J0 + 3,
                                          self.JD0)[:, :, 1:-2, :].reshape(
                                              B, T0, -1)
            elif self.J0 == 11:
                x2d_out = x2d_out.reshape(B, T0, self.J0 + 5,
                                          self.JD0)[:, :, 2:-3, :].reshape(
                                              B, T0, -1)
            x_out2d = self.linear_out(torch.concat([x_out1d, x2d_out], dim=2))
        return x_out1d, x_out2d

    def map2zq(self, x1d, x2d):
        N, T, _ = x1d.shape
        x_in = self.preprocess(x1d)
        x_encoder1d = self.encoder1d(x_in)
        code_idx1d, all_codes1d = self.quantizer1d.quantize(x_encoder1d,
                                                            return_latent=True)
        zq_x1d = self.quantizer1d.get_codes_from_indices(code_idx1d)

        B, T0, _, _ = x2d.shape
        if self.J0 == 30:
            x2d = torch.nn.functional.pad(x2d,
                                          (0, 0, 1, 1))  # for downsample X 2
        elif self.J0 == 13:
            x2d = torch.nn.functional.pad(x2d,
                                          (0, 0, 1, 2))  # for downsample X 2
        elif self.J0 == 11:
            x2d = torch.nn.functional.pad(x2d, (0, 0, 2, 3))
        x2d_encode = self.encoder2d(rearrange(x2d, 'b t j d -> b d t j'))
        B, D, T, _ = x2d_encode.shape
        code_idx2d, all_codes2d = self.quantizer2d.quantize(rearrange(
            x2d_encode, 'b d t j -> (b j) d t'),
                                                            return_latent=True)

        n_qt, _, _, _ = all_codes2d.shape
        code_idx2d = code_idx2d.reshape(B, self.J, T, n_qt).permute(0, 2, 1, 3)
        zq_x2d = self.quantizer2d.get_codes_from_indices(
            rearrange(code_idx2d, 'b t j n -> (b j) t n'))
        # print('zq_x1d.shape:', zq_x1d.shape)
        # print('code_idx1d.shape:', code_idx1d.shape)
        # print('code_idx2d.shape:', code_idx2d.shape)
        # print('zq_x2d.shape:', zq_x2d.shape)
        return zq_x1d, zq_x2d

    def map2latent(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        # print(x_encoder.shape)
        code_idx, all_codes = self.quantizer.quantize(x_encoder,
                                                      return_latent=True)
        x_d = self.quantizer.get_codes_from_indices(code_idx)
        x_out = x_d.sum(dim=0).permute(0, 2, 1)
        return x_out

    def map2index(self, x1d, x2d):
        N, T, _ = x1d.shape
        x_in = self.preprocess(x1d)
        x_encoder1d = self.encoder1d(x_in)
        code_idx1d, all_codes1d = self.quantizer1d.quantize(x_encoder1d,
                                                            return_latent=True)

        B, T0, _, _ = x2d.shape
        if self.J0 == 30:
            x2d = torch.nn.functional.pad(x2d,
                                          (0, 0, 1, 1))  # for downsample X 2
        elif self.J0 == 13:
            x2d = torch.nn.functional.pad(x2d,
                                          (0, 0, 1, 2))  # for downsample X 2
        elif self.J0 == 11:
            x2d = torch.nn.functional.pad(x2d, (0, 0, 2, 3))
        x2d_encode = self.encoder2d(rearrange(x2d, 'b t j d -> b d t j'))
        B, D, T, _ = x2d_encode.shape
        code_idx2d, all_codes2d = self.quantizer2d.quantize(rearrange(
            x2d_encode, 'b d t j -> (b j) d t'),
                                                            return_latent=True)
        n_qt, _, _, _ = all_codes2d.shape
        code_idx2d = code_idx2d.reshape(B, self.J, T, n_qt).permute(0, 2, 1, 3)

        return code_idx1d, code_idx2d


class RVQVAE_3D(nn.Module):

    def __init__(self,
                 args,
                 nb_code=256,
                 code_dim=256,
                 output_emb_width=256,
                 down_t_1d=2,
                 stride_t_1d=2,
                 down_s_1d=2,
                 stride_s_1d=0,
                 down_t_2d=3,
                 stride_t_2d=2,
                 width=256,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):

        super().__init__()
        self.code_dim1d = code_dim
        self.num_code1d = nb_code
        input_width = args.vae_test_dim
        s_input_width = 384
        self.encoder1d_t = Encoder(input_width,
                                   self.code_dim1d,
                                   down_t_1d,
                                   stride_t_1d,
                                   width,
                                   depth,
                                   dilation_growth_rate,
                                   activation=activation,
                                   norm=norm)
        self.decoder1d_t = Decoder(input_width,
                                   self.code_dim1d,
                                   down_t_1d,
                                   stride_t_1d,
                                   width,
                                   depth,
                                   dilation_growth_rate,
                                   activation=activation,
                                   norm=norm)
        self.encoder1d_s = Encoder_s(s_input_width,
                                     self.code_dim1d,
                                     down_t_1d,
                                     stride_t_1d,
                                     width,
                                     depth,
                                     dilation_growth_rate,
                                     activation=activation,
                                     norm=norm)
        self.decoder1d_s = Decoder_s(s_input_width,
                                     self.code_dim1d,
                                     down_s_1d,
                                     stride_s_1d,
                                     width,
                                     depth,
                                     dilation_growth_rate,
                                     activation=activation,
                                     norm=norm)

        rvqvae_config = {
            'num_quantizers': 6,
            'shared_codebook': False,
            'quantize_dropout_prob': 0.2,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim': code_dim,
            'args': args,
        }
        self.quantizer1d_s = ResidualVQ(**rvqvae_config)
        self.quantizer1d_t = ResidualVQ(**rvqvae_config)

        self.J0 = input_width // 6
        self.J = (self.J0 + 7) // 8
        self.JD0 = 6
        self.encode_dim2d = self.JD0
        self.num_code2d = nb_code  ###########
        self.code_dim2d = code_dim  ###########
        self.encoder2d = Encoder2d(self.encode_dim2d,
                                   self.code_dim2d,
                                   down_t_2d,
                                   stride_t_2d,
                                   width,
                                   depth,
                                   dilation_growth_rate,
                                   activation=activation,
                                   norm=norm)
        self.linear_merge = nn.Linear(self.code_dim2d * self.J,
                                      self.code_dim2d * self.J)
        self.decoder2d = Decoder2d(self.encode_dim2d,
                                   self.code_dim2d,
                                   down_t_2d,
                                   stride_t_2d,
                                   width,
                                   depth,
                                   dilation_growth_rate,
                                   activation=activation,
                                   norm=norm)
        self.linear_out_t = nn.Linear(input_width * 2, input_width)
        frame_dim = 64 * 6
        self.linear_out_s = nn.Linear(frame_dim * 2, frame_dim)
        joints_rvqvae_config = {
            'num_quantizers': 6,
            'shared_codebook': False,
            'quantize_dropout_prob': 0.2,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': self.num_code2d,
            'code_dim': self.code_dim2d,
            'args': args,
        }
        self.quantizer2d = ResidualVQ(**joints_rvqvae_config)

    def preprocess_t(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def preprocess_s(self, x):
        bs, n, j = x.shape
        x = x.reshape(bs, n, j // 6, 6)
        x = rearrange(x, 'b n j c -> b (n c) j')
        return x

    def forward(self, x1d, x2d):
        x_in_t = self.preprocess_t(x1d)
        # Encode
        x_encoder1d_t = self.encoder1d_t(x_in_t)
        ## quantization
        x_quantized1d_t, code_idx1d_t, commit_loss1d_t, perplexity1d_t = self.quantizer1d_t(
            x_encoder1d_t, sample_codebook_temp=0.5)
        ## decoder
        x_out1d_t = self.decoder1d_t(x_quantized1d_t)

        x_in_s = self.preprocess_s(x1d)
        x_encoder1d_s = self.encoder1d_s(x_in_s)
        x_quantized1d_s, code_idx1d_s, commit_loss1d_s, perplexity1d_s = self.quantizer1d_s(
            x_encoder1d_s, sample_codebook_temp=0.5)
        x_out1d_s = self.decoder1d_s(x_quantized1d_s)

        B, T0, _, _ = x2d.shape
        if self.J0 == 30:
            x2d = torch.nn.functional.pad(x2d,
                                          (0, 0, 1, 1))  # for downsample X 2
        elif self.J0 == 13:
            x2d = torch.nn.functional.pad(x2d,
                                          (0, 0, 1, 2))  # for downsample X 2
        elif self.J0 == 11:
            x2d = torch.nn.functional.pad(x2d, (0, 0, 2, 3))
        x2d_encode = self.encoder2d(rearrange(x2d, 'b t j d -> b d t j'))
        B, D, T, _ = x2d_encode.shape

        x2d_quantized, code_idx2d, commit_loss2d, perplexity2d = self.quantizer2d(
            rearrange(x2d_encode, 'b d t j -> (b j) d t'),
            sample_codebook_temp=0.5)
        x2d_quantized = x2d_quantized.reshape(B, self.J * D, T)
        x2d_quantized = self.linear_merge(
            x2d_quantized.permute(0, 2, 1).reshape(B * T, self.J * D)).reshape(
                B, T, self.J * D).permute(0, 2, 1)
        x2d_out = self.decoder2d(
            rearrange(x2d_quantized.reshape(B, self.J, D, T),
                      'b j d t -> b d t j'))

        T0 = x2d_out.shape[1]
        if self.J0 == 30:
            x2d_out = x2d_out.reshape(B, T0, self.J0 + 2,
                                      self.JD0)[:, :,
                                                1:-1, :].reshape(B, T0, -1)
        elif self.J0 == 13:
            x2d_out = x2d_out.reshape(B, T0, self.J0 + 3,
                                      self.JD0)[:, :,
                                                1:-2, :].reshape(B, T0, -1)
        elif self.J0 == 11:
            x2d_out = x2d_out.reshape(B, T0, self.J0 + 5,
                                      self.JD0)[:, :,
                                                2:-3, :].reshape(B, T0, -1)
        x_out2d_t = self.linear_out_t(torch.concat([x_out1d_t, x2d_out],
                                                   dim=2))
        x2d_out_s = rearrange(x2d_out, ('b t (j c) -> b j (t c)'), c=6)
        x_out2d_s = self.linear_out_s(
            torch.concat([x_out1d_s, x2d_out_s], dim=2))
        x_out2d_s = rearrange(x_out2d_s, ('b j (t c) -> b t (j c)'), c=6)
        x_out1d_s = rearrange(x_out1d_s, 'b j (t c) -> b t (j c)', c=6)
        x_out2d = (x_out2d_t + x_out2d_s) / 2
        output = {
            'rec_pose_1d_s': x_out1d_s,
            'rec_pose_1d_t': x_out1d_t,
            'rec_pose_2d': x_out2d,
            'embedding_loss_1d_s': commit_loss1d_s,
            'embedding_loss_1d_t': commit_loss1d_t,
            'embedding_loss_2d': commit_loss2d,
        }
        return output

    # def map2zp(self, x):
    #     N, T, _ = x.shape
    #     x_in = self.preprocess(x)
    #     x_encoder = self.encoder(x_in)
    #     return x_encoder

    def decode(self, x1d_s=None, x1d_t=None, x2d=None):
        x_out1d_s = None
        x_out1d_t = None
        if x1d_t is not None:
            x_d_t = self.quantizer1d_t.get_codes_from_indices(x1d_t)
            x1d_t = x_d_t.sum(dim=0).permute(0, 2, 1)
            x_out1d_t = self.decoder1d_t(x1d_t)
        if x1d_s is not None:

            num_clip = x1d_s.shape[1] // self.J0
            x_out1d_s = []
            for i in range(num_clip):
                x_d_s = self.quantizer1d_s.get_codes_from_indices(
                    x1d_s[:, i * self.J0:(i + 1) * self.J0, :])
                x_d_s = x_d_s.sum(dim=0).permute(0, 2, 1)
                x_out1d_s_sample = self.decoder1d_s(x_d_s)
                if i != num_clip - 1:
                    x_out1d_s.append(
                        (x_out1d_s_sample.reshape(1, self.J0, 64,
                                                  6))[:, :, :56, :])
                else:
                    x_out1d_s.append(
                        x_out1d_s_sample.reshape(1, self.J0, 64, 6))
            x_out1d_s = torch.cat(x_out1d_s, dim=2)

        x2d_out = None
        if x2d is not None:
            B, T, _, _ = x2d.shape
            x_d_2d = self.quantizer2d.get_codes_from_indices(
                rearrange(x2d, 'b t j n -> (b j) t n'))
            D = x_d_2d.shape[-1]
            x2d = rearrange(
                x_d_2d.sum(dim=0).reshape(B, self.J, T, D),
                'b j t d -> b (j d) t')
            x2d_quantized = self.linear_merge(
                x2d.permute(0, 2, 1).reshape(B * T, self.J * D)).reshape(
                    B, T, self.J * D).permute(0, 2, 1)
            x2d_out = self.decoder2d(
                rearrange(x2d_quantized.reshape(B, self.J, D, T),
                          'b j d t -> b d t j'))
            T0 = x2d_out.shape[1]
            if self.J0 == 30:
                x2d_out = x2d_out.reshape(B, T0, self.J0 + 2,
                                          self.JD0)[:, :, 1:-1, :].reshape(
                                              B, T0, -1)
            elif self.J0 == 13:
                x2d_out = x2d_out.reshape(B, T0, self.J0 + 3,
                                          self.JD0)[:, :, 1:-2, :].reshape(
                                              B, T0, -1)
            elif self.J0 == 11:
                x2d_out = x2d_out.reshape(B, T0, self.J0 + 5,
                                          self.JD0)[:, :, 2:-3, :].reshape(
                                              B, T0, -1)

        x_out2d_t = self.linear_out_t(torch.concat([x_out1d_t, x2d_out],
                                                   dim=2))
        # x2d_out_s = rearrange(x2d_out, ('b t (j c) -> b j (t c)'), c=6)
        t = min(x_out1d_s.shape[2], x2d_out.shape[1])
        num_clip_t = t // 64
        x_out1d_s = x_out1d_s[:, :, :num_clip_t * 64, :]
        x2d_out_s = x2d_out[:, :num_clip_t * 64, :]
        x2d_out_s = rearrange(x2d_out_s, ('b (n t) (j c) -> b n j (t c)'),
                              n=num_clip_t,
                              c=6)
        x_out1d_s = rearrange(x_out1d_s, ('b j (n t c) -> b n j (t c)'),
                              n=num_clip_t,
                              c=6)
        x_out2d_s = self.linear_out_s(
            torch.concat([x_out1d_s, x2d_out_s], dim=2))
        x_out2d_s = rearrange(x_out2d_s, ('b n j (t c) -> b (n t) (j c)'), c=6)
        x_out1d_s = rearrange(x_out1d_s, 'b n j (t c) -> b (n t) (j c)', c=6)
        x_out2d_t = x_out2d_t[:, :num_clip_t * 64, :]
        x_out2d = (x_out2d_t + x_out2d_s) / 2
        return x_out1d_s, x_out1d_t, x_out2d

    def map2zq(self, x1d, x2d):
        N, T, _ = x1d.shape
        x_in_s = self.preprocess_s(x1d)
        x_encoder1d_s = self.encoder1d_s(x_in_s)
        code_idx1d_s, all_codes1d_s = self.quantizer1d_s.quantize(
            x_encoder1d_s, return_latent=True)
        zq_x1d_s = self.quantizer1d_s.get_codes_from_indices(code_idx1d_s)

        x_in_t = self.preprocess_t(x1d)
        x_encoder1d_t = self.encoder1d_t(x_in_t)
        code_idx1d_t, all_codes1d_t = self.quantizer1d_t.quantize(
            x_encoder1d_t, return_latent=True)
        zq_x1d_t = self.quantizer1d_t.get_codes_from_indices(code_idx1d_t)

        B, T0, _, _ = x2d.shape
        if self.J0 == 30:
            x2d = torch.nn.functional.pad(x2d,
                                          (0, 0, 1, 1))  # for downsample X 2
        elif self.J0 == 13:
            x2d = torch.nn.functional.pad(x2d,
                                          (0, 0, 1, 2))  # for downsample X 2
        elif self.J0 == 11:
            x2d = torch.nn.functional.pad(x2d, (0, 0, 2, 3))
        x2d_encode = self.encoder2d(rearrange(x2d, 'b t j d -> b d t j'))
        B, D, T, _ = x2d_encode.shape
        code_idx2d, all_codes2d = self.quantizer2d.quantize(rearrange(
            x2d_encode, 'b d t j -> (b j) d t'),
                                                            return_latent=True)

        n_qt, _, _, _ = all_codes2d.shape
        code_idx2d = code_idx2d.reshape(B, self.J, T, n_qt).permute(0, 2, 1, 3)
        zq_x2d = self.quantizer2d.get_codes_from_indices(
            rearrange(code_idx2d, 'b t j n -> (b j) t n'))

        return zq_x1d_s, zq_x1d_t, zq_x2d

    # def map2latent(self, x):
    #     N, T, _ = x.shape
    #     x_in = self.preprocess(x)
    #     x_encoder = self.encoder(x_in)
    #     # print(x_encoder.shape)
    #     code_idx, all_codes = self.quantizer.quantize(x_encoder,
    #                                                   return_latent=True)
    #     x_d = self.quantizer.get_codes_from_indices(code_idx)
    #     x_out = x_d.sum(dim=0).permute(0, 2, 1)
    #     return x_out

    def map2index(self, x1d, x2d):
        N, T, _ = x1d.shape
        x_in_s = self.preprocess_s(x1d)
        x_encoder1d_s = self.encoder1d_s(x_in_s)
        code_idx1d_s, all_codes1d_s = self.quantizer1d_s.quantize(
            x_encoder1d_s, return_latent=True)

        x_in_t = self.preprocess_t(x1d)
        x_encoder1d_t = self.encoder1d_t(x_in_t)
        code_idx1d_t, all_codes1d_t = self.quantizer1d_t.quantize(
            x_encoder1d_t, return_latent=True)

        B, T0, _, _ = x2d.shape
        if self.J0 == 30:
            x2d = torch.nn.functional.pad(x2d,
                                          (0, 0, 1, 1))  # for downsample X 2
        elif self.J0 == 13:
            x2d = torch.nn.functional.pad(x2d,
                                          (0, 0, 1, 2))  # for downsample X 2
        elif self.J0 == 11:
            x2d = torch.nn.functional.pad(x2d, (0, 0, 2, 3))
        x2d_encode = self.encoder2d(rearrange(x2d, 'b t j d -> b d t j'))
        B, D, T, _ = x2d_encode.shape
        code_idx2d, all_codes2d = self.quantizer2d.quantize(rearrange(
            x2d_encode, 'b d t j -> (b j) d t'),
                                                            return_latent=True)
        n_qt, _, _, _ = all_codes2d.shape
        code_idx2d = code_idx2d.reshape(B, self.J, T, n_qt).permute(0, 2, 1, 3)

        return code_idx1d_s, code_idx1d_t, code_idx2d


class RVQVAE(nn.Module):

    def __init__(self,
                 args,
                 nb_code=256,
                 code_dim=256,
                 output_emb_width=256,
                 down_t=2,
                 stride_t=2,
                 width=256,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):

        super().__init__()
        assert output_emb_width == code_dim
        input_width = args.vae_test_dim  # 180
        # print('input_width:',input_width.shape)
        # exit()
        self.code_dim = code_dim
        self.num_code = nb_code

        # self.quant = args.quantizer
        self.encoder = Encoder(input_width,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)
        self.decoder = Decoder(input_width,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)
        # self.encoder = tDEncoder(6, output_emb_width, down_t, down_t, stride_t,  stride_t, width, depth,
        #    dilation_growth_rate, activation=activation, norm=norm)

        # self.decoder = tDDecoder(output_emb_width, 6, down_t, down_t, stride_t, stride_t, width, depth,
        #    dilation_growth_rate, activation=activation, norm=norm)
        # self.decoder = Causal_Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
        #                        dilation_growth_rate, activation=activation, norm=norm)
        rvqvae_config = {
            'num_quantizers': 6,
            'shared_codebook': False,
            'quantize_dropout_prob': 0.2,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim': code_dim,
            'args': args,
        }
        self.quantizer = ResidualVQ(**rvqvae_config)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        code_idx, all_codes = self.quantizer.quantize(x_encoder,
                                                      return_latent=True)
        # print(code_idx.shape)
        # code_idx = code_idx.view(N, -1)
        # (N, T, Q)
        # print()
        return code_idx, all_codes

    def forward(self, x):
        # print('x:', x.shape)
        x_in = self.preprocess(x)
        # print('x_in:', x_in.shape)
        x_encoder = self.encoder(x_in)
        # print('x_encoder:', x_encoder.shape)
        # exit()
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(
            x_encoder, sample_codebook_temp=0.5)
        # print('x_quantized', x_quantized.shape)
        # print('code_idx shape', code_idx.shape)
        # print('code_idx:', code_idx[0, :, 1])
        # exit()
        if torch.isnan(x_quantized).any() or torch.isinf(x_quantized).any():
            print("NaN or Inf detected in quantized encoding")

        if torch.isnan(commit_loss).any() or torch.isinf(commit_loss).any():
            print("NaN or Inf detected in commit loss")
        # print('x_quantized:', x_quantized.shape)
        # exit()
        x_out = self.decoder(x_quantized)
        # print('x_out:', x_out.shape)
        # exit()
        output = {
            'rec_pose': x_out,
            'embedding_loss': commit_loss,
        }
        # code_idx, all_codes = self.encode(x)
        # x_out_index = self.forward_decoder(code_idx)
        # print('x_out:', x_out.shape)
        # print('x_out_index:', x_out_index.shape)
        # print(sum(sum(x_out == x_out_index)))
        # exit()
        return output

    def map2zp(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        return x_encoder

    def decode(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x = x_d.sum(dim=0).permute(0, 2, 1)
        # print('x:', x.shape)
        # exit()
        # decoder
        x_out = self.decoder(x)
        # x_out = self.postprocess(x_decoder)
        return x_out

    def map2zq(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        # print(x_encoder.shape)
        # exit()
        code_idx, all_codes = self.quantizer.quantize(x_encoder,
                                                      return_latent=True)
        x_d = self.quantizer.get_codes_from_indices(code_idx)
        return x_d

    def map2latent(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        # print(x_encoder.shape)
        code_idx, all_codes = self.quantizer.quantize(x_encoder,
                                                      return_latent=True)
        x_d = self.quantizer.get_codes_from_indices(code_idx)
        x_out = x_d.sum(dim=0).permute(0, 2, 1)
        return x_out

    def map2index(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        # print(x_encoder.shape)
        code_idx, all_codes = self.quantizer.quantize(x_encoder,
                                                      return_latent=True)
        return code_idx
