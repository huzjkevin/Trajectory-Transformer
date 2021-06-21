"""
    This python file is to make an alternative to individual_TF that use pytorch's bulit-in functions and classes
    to implement transformer for trajectory prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io
import os

import copy
import math


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


class IndividualTF(nn.Module):
    def __init__(
        self,
        enc_inp_size,
        dec_inp_size,
        dec_out_size,
        N=6,
        d_model=512,
        d_ff=2048,
        h=8,
        dropout=0.1,
        mean=[0, 0],
        std=[0, 0],
        noise_dim=16,
    ):
        super(IndividualTF, self).__init__()
        "Helper: Construct a model from hyperparameters."

        self.transformer = torch.nn.Transformer(
            d_model=d_model,
            nhead=h,
            num_encoder_layers=N,
            num_decoder_layers=6,
            dim_feedforward=d_ff,
        )

        self.generator = nn.Linear(d_model, dec_out_size)
        self.src_embbeding = nn.Sequential(
            LinearEmbedding(enc_inp_size, d_model), PositionalEncoding(d_model, dropout)
        )
        self.tgt_embedding = nn.Sequential(
            LinearEmbedding(dec_inp_size, d_model), PositionalEncoding(d_model, dropout)
        )

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        # for p in self.model.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask, seq_start_end=None):
        src_emb = self.src_embbeding(src.permute(1, 0, 2))
        tgt_emb = self.tgt_embedding(tgt.permute(1, 0, 2))
        output = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask)

        return self.generator(output.permute(1, 0, 2))


class LinearEmbedding(nn.Module):
    def __init__(self, inp_size, d_model):
        super(LinearEmbedding, self).__init__()
        # lut => lookup table
        self.lut = nn.Linear(inp_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, out_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, out_size)

    def forward(self, x):
        return self.proj(x)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

    return src_mask, tgt_mask
