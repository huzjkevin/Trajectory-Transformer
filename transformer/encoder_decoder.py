# -*- coding: utf-8 -*-
# date: 2018-11-29 19:56
import torch
import torch.nn as nn


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, noise_dim=16):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.noise_dim = noise_dim

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Take in and process masked src and target sequences.
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        feat_before_noise = self.encoder(self.src_embed(src), src_mask)
        n_batch, seq_len, _ = feat_before_noise.size()
        noise = torch.randn((n_batch, seq_len, self.noise_dim)).cuda()
        feat = torch.cat((feat_before_noise, noise), dim=-1)
        return feat

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
