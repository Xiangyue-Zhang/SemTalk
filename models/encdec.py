import torch.nn as nn
from .resnet import Resnet1D, Resnet2D
import torch
import torch.nn.functional as F


class Decoder2d(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []

        blocks.append(nn.Conv2d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet2D(width,
                         depth,
                         dilation_growth_rate,
                         reverse_dilation=True,
                         activation=activation,
                         norm=norm), nn.Upsample(scale_factor=2,
                                                 mode='nearest'),
                nn.Conv2d(width, out_dim, 3, 1, 1))
            blocks.append(block)
        blocks.append(nn.Conv2d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv2d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.model(x)
        return x.permute(0, 2, 3, 1)


class Encoder2d(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv2d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv2d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet2D(width,
                         depth,
                         dilation_growth_rate,
                         activation=activation,
                         norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv2d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)
        self.blocks = blocks

    def forward(self, x):
        return self.model(x)


# class Encoder(nn.Module):
#     def __init__(self,
#                  input_emb_width=3,
#                  output_emb_width=512,
#                  down_t=2,
#                  stride_t=2,
#                  width=512,
#                  depth=3,
#                  dilation_growth_rate=3,
#                  activation='relu',
#                  norm=None):
#         super().__init__()

#         blocks = []
#         filter_t, pad_t = stride_t * 2, stride_t // 2
#         blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
#         blocks.append(nn.ReLU())

#         for i in range(down_t):
#             input_dim = width
#             block = nn.Sequential(
#                 # nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
#                 nn.Conv1d(input_dim, width, 3, 1, 1),

#                 Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
#             )
#             blocks.append(block)
#         blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
#         self.model = nn.Sequential(*blocks)

#     def forward(self, x):
#         output = self.model(x)
#         if torch.isnan(output).any() or torch.isinf(output).any():
#             print("NaN or Inf detected in Encoder output")
#         return output


class PaddingLayer(nn.Module):

    def __init__(self, target_multiple=4):
        super().__init__()
        self.target_multiple = target_multiple

    def forward(self, x):
        """
        Args:
            x: (B, 6, J, T)
        Returns:
            padded_x: (B, 6, J_padded, T)
            pad: (left_pad, right_pad)
        """
        B, C, J, T = x.shape
        remainder = J % self.target_multiple
        if remainder == 0:
            return x, (0, 0)

        pad = self.target_multiple - remainder
        left_pad = pad // 2
        right_pad = pad - left_pad
        padded_x = F.pad(x, (0, 0, left_pad, right_pad))  # Pad along J
        return padded_x, (left_pad, right_pad)


class tDEncoder_noj(nn.Module):

    def __init__(
            self,
            input_emb_width=6,  # 每个关节的rot6d
            output_emb_width=512,
            down_t=2,  # 时间维度的下采样次数
            down_j=2,  # 关节维度的下采样次数
            stride_t=2,
            stride_j=2,
            width=512,
            depth=3,
            dilation_growth_rate=3,
            activation='relu',
            norm=None):
        super().__init__()

        blocks = []
        # self.padding_layer = PaddingLayer(target_multiple=4)

        # 初始二维卷积
        kernel_j = 3 if down_j > 0 else 1
        stride_j_actual = stride_j if down_j > 0 else 1
        blocks.append(
            nn.Conv2d(input_emb_width,
                      width,
                      kernel_size=(kernel_j, 3),
                      stride=(stride_j_actual, 1),
                      padding=(kernel_j // 2, 1)))
        blocks.append(nn.ReLU())

        # 多层下采样
        for _ in range(max(down_t, down_j)):
            kernel_j = 3 if down_j > 0 else 1
            stride_j_actual = stride_j if down_j > 0 else 1
            kernel_t = 3 if down_t > 0 else 1
            stride_t_actual = stride_t if down_t > 0 else 1
            block = nn.Sequential(
                Resnet2D(width,
                         depth,
                         dilation_growth_rate,
                         activation=activation,
                         norm=norm),
                nn.Conv2d(width,
                          width,
                          kernel_size=(kernel_j, kernel_t),
                          stride=(stride_j_actual, stride_t_actual),
                          padding=(kernel_j // 2, kernel_t // 2)))
            blocks.append(block)
            down_j = max(0, down_j - 1)
            down_t = max(0, down_t - 1)

        # 输出卷积
        kernel_j = 3 if down_j > 0 else 1
        blocks.append(
            nn.Conv2d(width,
                      output_emb_width,
                      kernel_size=(kernel_j, 3),
                      stride=(1, 1),
                      padding=(kernel_j // 2, 1)))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        """
        Args:
            x: (B, T, C)
        Returns:
            encoded_x: (B, output_emb_width, J_down, T_down)
            pad: (left_pad, right_pad)
        """
        B, C, T = x.shape
        J = C // 6  # 关节数
        x = x.view(B, J, 6, T)  # 重组为 (B, J, 6, T)
        x = x.permute(0, 2, 1, 3)  # 转为 (B, 6, J, T)

        # Padding
        # x, pad = self.padding_layer(x)  # x: (B, 6, J_padded, T)

        # 编码
        encoded_x = self.model(x)

        return encoded_x


class CroppingLayer(nn.Module):

    def forward(self, x, pad):
        """
        Args:
            x: (B, C, J_padded, T)
            pad: (left_pad, right_pad)
        Returns:
            cropped_x: (B, C, J, T)
        """
        left_pad, right_pad = pad
        if right_pad > 0:
            return x[:, :, left_pad:-right_pad, :]
        return x[:, :, left_pad:, :]


class tDDecoder_noj(nn.Module):

    def __init__(
            self,
            input_emb_width=512,  # 与编码器的输出相同
            output_emb_width=6,  # rot6d 的特征维度
            up_t=2,  # 时间维度的上采样次数
            up_j=2,  # 关节维度的上采样次数
            stride_t=2,
            stride_j=2,
            width=512,
            depth=3,
            dilation_growth_rate=3,
            activation='relu',
            norm=None):
        super().__init__()
        blocks = []

        # self.cropping_layer = CroppingLayer()

        # 初始卷积：将输入的低维特征映射到更高维
        kernel_j = 3 if up_j > 0 else 1
        stride_j_actual = stride_j if up_j > 0 else 1
        blocks.append(
            nn.Conv2d(input_emb_width,
                      width,
                      kernel_size=(kernel_j, 3),
                      stride=(stride_j_actual, 1),
                      padding=(kernel_j // 2, 1)))
        blocks.append(nn.ReLU())

        # 多层上采样
        for _ in range(max(up_t, up_j)):
            kernel_j = 3 if up_j > 0 else 1
            stride_j_actual = stride_j if up_j > 0 else 1
            kernel_t = 3 if up_t > 0 else 1
            stride_t_actual = stride_t if up_t > 0 else 1
            block = nn.Sequential(
                Resnet2D(width,
                         depth,
                         dilation_growth_rate,
                         reverse_dilation=True,
                         activation=activation,
                         norm=norm),
                nn.Upsample(scale_factor=(stride_j_actual, stride_t_actual),
                            mode='nearest'),
                nn.Conv2d(width,
                          width,
                          kernel_size=(kernel_j, kernel_t),
                          stride=(1, 1),
                          padding=(kernel_j // 2, kernel_t // 2)))
            blocks.append(block)
            up_j = max(0, up_j - 1)
            up_t = max(0, up_t - 1)

        # 输出卷积：将高维特征映射回 rot6d
        kernel_j = 3 if up_j > 0 else 1
        blocks.append(
            nn.Conv2d(width,
                      output_emb_width,
                      kernel_size=(kernel_j, 3),
                      stride=(1, 1),
                      padding=(kernel_j // 2, 1)))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        """
        Args:
            x: (B, input_emb_width, J_down, T_down)
            pad: (left_pad, right_pad)
        Returns:
            x: (B, T, J * rot6d)
        """
        # 解码器模型
        x = self.model(x)  # (B, output_emb_width, J_padded, T)

        # 裁剪多余的关节
        # x = self.cropping_layer(x, pad)  # (B, output_emb_width, J, T)

        # 调整维度回到原始格式
        x = x.permute(0, 2, 3, 1)  # 转为 (B, J, T, output_emb_width)
        B, J, T, rot6d = x.shape
        x = x.reshape(B, T, J * rot6d)  # 重组为 (B, T, J * rot6d)
        return x


class tDEncoder(nn.Module):

    def __init__(
            self,
            input_emb_width=6,  # 每个关节的rot6d
            output_emb_width=512,
            down_t=2,  # 时间维度的下采样次数
            down_j=2,  # 关节维度的下采样次数
            stride_t=2,
            stride_j=2,
            width=512,
            depth=3,
            dilation_growth_rate=3,
            activation='relu',
            norm=None):
        super().__init__()

        blocks = []
        filter_t, filter_j = stride_t * 2, stride_j * 2
        pad_t, pad_j = stride_t // 2, stride_j // 2

        self.padding_layer = PaddingLayer(target_multiple=4)

        # 初始二维卷积
        blocks.append(
            nn.Conv2d(input_emb_width,
                      width,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)))
        blocks.append(nn.ReLU())

        # 多层下采样
        for _ in range(max(down_t, down_j)):
            block = nn.Sequential(
                nn.Conv2d(width,
                          width,
                          kernel_size=(filter_j, filter_t),
                          stride=(stride_j if down_j > 0 else 1,
                                  stride_t if down_t > 0 else 1),
                          padding=(pad_j, pad_t)),
                Resnet2D(width,
                         depth,
                         dilation_growth_rate,
                         activation=activation,
                         norm=norm),
            )
            blocks.append(block)
            down_j = max(0, down_j - 1)
            down_t = max(0, down_t - 1)

        # 输出卷积
        blocks.append(
            nn.Conv2d(width,
                      output_emb_width,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        """
        Args:
            x: (B, T, C)
        Returns:
            encoded_x: (B, output_emb_width, J_down, T_down)
            pad: (left_pad, right_pad)
        """
        B, C, T = x.shape
        J = C // 6  # 关节数
        x = x.view(B, J, 6, T)  # 重组为 (B, J, 6, T)
        x = x.permute(0, 2, 1, 3)  # 转为 (B, 6, J, T)

        # Padding
        x, pad = self.padding_layer(x)  # x: (B, 6, J_padded, T)

        # 编码
        encoded_x = self.model(x)

        return encoded_x, pad


class tDDecoder(nn.Module):

    def __init__(
            self,
            input_emb_width=512,  # 与编码器的输出相同
            output_emb_width=6,  # rot6d 的特征维度
            up_t=2,  # 时间维度的上采样次数
            up_j=2,  # 关节维度的上采样次数
            stride_t=2,
            stride_j=2,
            width=512,
            depth=3,
            dilation_growth_rate=3,
            activation='relu',
            norm=None):
        super().__init__()
        blocks = []

        self.cropping_layer = CroppingLayer()

        # 初始卷积：将输入的低维特征映射到更高维
        blocks.append(
            nn.Conv2d(input_emb_width,
                      width,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)))
        blocks.append(nn.ReLU())

        # 上采样块
        for _ in range(max(up_t, up_j)):
            block = nn.Sequential(
                Resnet2D(width,
                         depth,
                         dilation_growth_rate,
                         reverse_dilation=True,
                         activation=activation,
                         norm=norm),
                nn.Upsample(scale_factor=(stride_j if up_j > 0 else 1,
                                          stride_t if up_t > 0 else 1),
                            mode='nearest'),
                nn.Conv2d(width,
                          width,
                          kernel_size=(3, 3),
                          stride=(1, 1),
                          padding=(1, 1)))
            blocks.append(block)
            up_j = max(0, up_j - 1)
            up_t = max(0, up_t - 1)

        # 输出卷积：将高维特征映射回 rot6d
        blocks.append(
            nn.Conv2d(width,
                      output_emb_width,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=(1, 1)))
        self.model = nn.Sequential(*blocks)

    def forward(self, x, pad):
        """
        Args:
            x: (B, input_emb_width, J_down, T_down)
            pad: (left_pad, right_pad)
        Returns:
            x: (B, T, J * rot6d)
        """
        # 解码器模型
        x = self.model(x)  # (B, output_emb_width, J_padded, T)

        # 裁剪多余的关节
        x = self.cropping_layer(x, pad)  # (B, output_emb_width, J, T)

        # 调整维度回到原始格式
        x = x.permute(0, 2, 3, 1)  # 转为 (B, J, T, output_emb_width)
        B, J, T, rot6d = x.shape
        x = x.reshape(B, T, J * rot6d)  # 重组为 (B, T, J * rot6d)
        return x


class Encoder(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                # nn.Conv1d(input_dim, width, 3, 1, 1),
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         activation=activation,
                         norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Encoder_s(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        # filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, 3, 1, 1),
                # nn.Conv1d(input_dim, width, 3, 1, 1),
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         activation=activation,
                         norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


# class Decoder(nn.Module):
#     def __init__(self,
#                  input_emb_width=3,
#                  output_emb_width=512,
#                  down_t=2,
#                  stride_t=2,
#                  width=512,
#                  depth=3,
#                  dilation_growth_rate=3,
#                  activation='relu',
#                  norm=None):
#         super().__init__()
#         blocks = []

#         blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
#         blocks.append(nn.ReLU())
#         for i in range(down_t):
#             out_dim = width
#             block = nn.Sequential(
#                 Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
#                 # nn.Upsample(scale_factor=2, mode='nearest'),
#                 nn.Conv1d(width, out_dim, 3, 1, 1)
#             )
#             blocks.append(block)
#         blocks.append(nn.Conv1d(width, width, 3, 1, 1))
#         blocks.append(nn.ReLU())
#         blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
#         self.model = nn.Sequential(*blocks)

#     def forward(self, x):
#         output = self.model(x)
#         if torch.isnan(output).any() or torch.isinf(output).any():
#             print("NaN or Inf detected in Decoder output")
#         return output.permute(0, 2, 1)


class Decoder(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []

        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         reverse_dilation=True,
                         activation=activation,
                         norm=norm), nn.Upsample(scale_factor=2,
                                                 mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1))
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.model(x)
        return x.permute(0, 2, 1)


class Decoder_s(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []

        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         reverse_dilation=True,
                         activation=activation,
                         norm=norm),
                # nn.Upsample(scale_factor=2,mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1))
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.model(x)
        return x.permute(0, 2, 1)


class Causal_Decoder(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []

        kernel_size = 3
        padding = (kernel_size - 1)  # 因果填充

        blocks.append(
            nn.Conv1d(output_emb_width,
                      width,
                      kernel_size,
                      stride=1,
                      padding=padding))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         reverse_dilation=True,
                         activation=activation,
                         norm=norm),
                # nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width,
                          out_dim,
                          kernel_size,
                          stride=1,
                          padding=padding))
            blocks.append(block)
        blocks.append(
            nn.Conv1d(width, width, kernel_size, stride=1, padding=padding))
        blocks.append(nn.ReLU())
        blocks.append(
            nn.Conv1d(width,
                      input_emb_width,
                      kernel_size,
                      stride=1,
                      padding=padding))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.model(x)
        return x.permute(0, 2, 1)
