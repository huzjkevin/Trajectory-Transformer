# -*- coding: utf-8 -*-
# date: 2018-11-29 19:56
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder1, encoder2, decoder, src_embed, tgt_embed, generator, noise_dim=16):
        super(EncoderDecoder, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.noise_dim = noise_dim


    def forward(self, src, tgt, src_mask, tgt_mask, seq_start_end):
        """
        Take in and process masked src and target sequences.
        """
        return self.decode(
            self.encode(src, src_mask, seq_start_end), src_mask, tgt, tgt_mask
        )

    def encode(self, src, src_mask, seq_start_end):
        # feat = self.encoder1(self.src_embed(src), src_mask, seq_start_end)
        feat = self.encoder2(self.src_embed(src), src_mask)
        return feat
        # return self.encoder(self.src_embed(src), src_mask, seq_start_end)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)



