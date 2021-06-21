# -*- coding: utf-8 -*-
# date: 2018-11-29 19:56
import torch
import torch.nn as nn


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    # def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, noise_dim=16):
    #     super(EncoderDecoder, self).__init__()
    #     self.encoder = encoder
    #     self.decoder = decoder
    #     self.src_embed = src_embed
    #     self.tgt_embed = tgt_embed
    #     self.generator = generator
    #     self.noise_dim = noise_dim

    def __init__(self, transformer, src_embed, tgt_embed, generator, noise_dim=16):
        super(EncoderDecoder, self).__init__()
        self.transformer = transformer
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.noise_dim = noise_dim

    def forward(self, src, tgt, src_mask, tgt_mask, seq_start_end):
        """
        Take in and process masked src and target sequences.
        """
        # return self.decode(self.encode(src, src_mask, seq_start_end), src_mask, tgt, tgt_mask)
        enc_emb = self.src_embed(src).permute(1, 0, 2)
        dec_emb = self.tgt_embed(tgt).permute(1, 0, 2)
        return self.transformer(enc_emb, dec_emb, src_mask, tgt_mask).permute(1, 0, 2)

    def encode(self, src, src_mask, seq_start_end):
        return self.encoder(self.src_embed(src), src_mask, seq_start_end)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
