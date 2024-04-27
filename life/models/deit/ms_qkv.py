import math
import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, padding_mode, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   groups=in_channels, bias=bias, padding=padding, padding_mode=padding_mode)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class MultiscaleQKV(nn.Module):
    def __init__(self, dim, qkv_dim, cls_token=True, concat_qkv=False, norm=None):
        super(MultiscaleQKV, self).__init__()
        self.dim = dim
        self.qkv_dim = qkv_dim
        self.cls_token = cls_token
        self.concat_qkv = concat_qkv
        self.scale1 = nn.Linear(dim, qkv_dim * 3)  # This is equivalent to 1 x 1 convolution
        self.scale2 = SeparableConv2d(in_channels=dim, out_channels=qkv_dim * 3, kernel_size=3, padding=1,
                                      padding_mode='zeros')
        self.scale3 = SeparableConv2d(in_channels=dim, out_channels=qkv_dim * 3, kernel_size=5, padding=2,
                                      padding_mode='zeros')
        self.norm = norm

    def forward(self, x):
        if self.cls_token:
            x = self.forward_with_cls_token(x)
        else:
            x = self.forward_wo_cls_token(x)

        return x

    def forward_wo_cls_token(self, x):
        B, P, C = x.shape

        # scale 1
        scale1_qkv = self.scale1(x)

        # preprocess for scale 2 and 3
        num_patches = P
        fmap_shape = int(math.sqrt(num_patches))  # feature map shape
        x = scale1_qkv.permute((0, 2, 1)).reshape(
            (B, C, fmap_shape, fmap_shape))  # reshaping as feature maps for conv layers

        # scale 2
        scale2_qkv = self.scale2(x)

        # scale 3
        scale3_qkv = self.scale3(scale2_qkv)

        # post process for scale 2 and 3
        scale2_qkv = scale2_qkv.view((B, C, -1)).permute((0, 2, 1))
        scale3_qkv = scale3_qkv.view((B, C, -1)).permute((0, 2, 1))

        q = torch.cat((scale1_qkv[:, :, :self.qkv_dim],
                       scale2_qkv[:, :, :self.qkv_dim],
                       scale3_qkv[:, :, :self.qkv_dim]),
                      dim=2)

        k = torch.cat((scale1_qkv[:, :, self.qkv_dim:self.qkv_dim * 2],
                       scale2_qkv[:, :, self.qkv_dim:self.qkv_dim * 2],
                       scale3_qkv[:, :, self.qkv_dim:self.qkv_dim * 2]),
                      dim=2)

        v = torch.cat((scale1_qkv[:, :, self.qkv_dim * 2:],
                       scale2_qkv[:, :, self.qkv_dim * 2:],
                       scale3_qkv[:, :, self.qkv_dim * 2:]),
                      dim=2)

        return q, k, v

    def forward_with_cls_token(self, x):
        B, P, C = x.shape
        num_aux_tokens = 1
        num_patches = P - num_aux_tokens  # number of patches without class token
        fmap_shape = int(math.sqrt(num_patches))

        # scale 1
        scale1_qkv_ = self.scale1(x)

        # patch_tokens
        scale1_qkv = scale1_qkv_[:, num_aux_tokens:, :]  # all tokens without aux token
        x = scale1_qkv.permute((0, 2, 1)).reshape(
            (B, C, fmap_shape, fmap_shape))  # reshaping as feature maps for conv layers

        scale2_qkv = self.scale2(x)
        scale3_qkv = self.scale3(scale2_qkv)


        # aux_tokens
        scale1_aux_token = scale1_qkv_[:, :num_aux_tokens, :].view((B, C, -1, 1))
        scale2_aux_token = self.scale2.pointwise(scale1_aux_token)
        scale3_aux_token = self.scale3.pointwise(scale2_aux_token)

        # post process for scale 2 and 3
        scale2_qkv = scale2_qkv.view((B, C, -1)).permute((0, 2, 1))
        scale3_qkv = scale3_qkv.view((B, C, -1)).permute((0, 2, 1))
        scale1_aux_token = scale1_aux_token.reshape(B, num_aux_tokens, C)
        scale2_aux_token = scale2_aux_token.reshape(B, num_aux_tokens, C)
        scale3_aux_token = scale3_aux_token.reshape(B, num_aux_tokens, C)

        patch_token_q = torch.cat((scale1_qkv[:, :, :self.qkv_dim],
                                   scale2_qkv[:, :, :self.qkv_dim],
                                   scale3_qkv[:, :, :self.qkv_dim]), dim=2)
        aus_token_q = torch.cat((scale1_aux_token[:, :, :self.qkv_dim],
                                 scale2_aux_token[:, :, :self.qkv_dim],
                                 scale3_aux_token[:, :, :self.qkv_dim]), dim=2)
        q = torch.cat((aus_token_q, patch_token_q), dim=1)

        patch_token_k = torch.cat((scale1_qkv[:, :, self.qkv_dim:self.qkv_dim * 2],
                                   scale2_qkv[:, :, self.qkv_dim:self.qkv_dim * 2],
                                   scale3_qkv[:, :, self.qkv_dim:self.qkv_dim * 2]), dim=2)
        aux_token_k = torch.cat((scale1_aux_token[:, :, self.qkv_dim:self.qkv_dim * 2],
                                 scale1_aux_token[:, :, self.qkv_dim:self.qkv_dim * 2],
                                 scale1_aux_token[:, :, self.qkv_dim:self.qkv_dim * 2]), dim=2)
        k = torch.cat((aux_token_k, patch_token_k), dim=1)

        patch_token_v = torch.cat((scale1_qkv[:, :, self.qkv_dim * 2:],
                                   scale2_qkv[:, :, self.qkv_dim * 2:],
                                   scale3_qkv[:, :, self.qkv_dim * 2:]), dim=2)
        aux_token_v = torch.cat((scale1_aux_token[:, :, self.qkv_dim * 2:],
                                 scale2_aux_token[:, :, self.qkv_dim * 2:],
                                 scale3_aux_token[:, :, self.qkv_dim * 2:]), dim=2)
        v = torch.cat((aux_token_v, patch_token_v), dim=1)
        if self.norm is not None:
            q = self.norm(q)
            k = self.norm(k)
            v = self.norm(v)

        if self.concat_qkv:
            return torch.cat((q, k, v), dim=2)
        return q, k, v