# -*- coding: utf-8 -*-
# date: 2018-11-29 20:07
import torch
import torch.nn as nn
import torch.nn.functional as F

from .functional import clones
from .layer_norm import LayerNorm


def make_mlp(dim_list, activation="relu", batch_norm=False, dropout=0, alpha=0.2):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "leakyrelu":
            layers.append(nn.LeakyReLU(negative_slope=alpha))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)

class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers
    """

    def __init__(self, layer, n):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

        # n_units = [512, 16, 512]
        # n_heads = [4, 1]
        # dropout = 0.2
        # alpha = 0.2
        # self.gatencoder = GATEncoder(
        #     n_units=n_units, n_heads=n_heads, dropout=dropout, alpha=alpha
        # )

    def forward(self, x, x_mask):
        """
        Pass the input (and mask) through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, x_mask)

        x = self.norm(x)

        # if seq_start_end is not None:
        #     x = self.gatencoder(x.permute(1, 0, 2), seq_start_end).permute(1, 0, 2)

        # return self.norm(x)
        return x


# class EncoderVer2(nn.Module):
#     """
#     Core encoder is a stack of N layers
#     This is a second version that has a Linear layer to compress and upcompress GAT channel
#     """

#     def __init__(self, layer, n):
#         super(EncoderVer2, self).__init__()
#         self.layers = clones(layer, n)
#         self.norm = LayerNorm(layer.size)
#         self.compress = make_mlp([512, 128, 32])
#         self.upcompress = make_mlp([32, 128, 512])

#         n_units = [32, 16, 32]
#         n_heads = [4, 1]
#         dropout = 0.2
#         alpha = 0.2
#         self.gatencoder = GATEncoder(
#             n_units=n_units, n_heads=n_heads, dropout=dropout, alpha=alpha
#         )

#     def forward(self, x, x_mask, seq_start_end=None):
#         """
#         Pass the input (and mask) through each layer in turn.
#         """
#         for layer in self.layers:
#             x = layer(x, x_mask)

#         x = self.norm(x)
        
#         if seq_start_end is not None:
#             x = torch.cat([self.compress(x[:, i, :]).unsqueeze(1) for i in range(x.size(1))], dim=1)
#             x = self.gatencoder(x.permute(1, 0, 2), seq_start_end).permute(1, 0, 2)
#             x = torch.cat([self.upcompress(x[:, i, :]).unsqueeze(1) for i in range(x.size(1))], dim=1)

#         # return self.norm(x)
#         return x

