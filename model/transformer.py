"""
 * @Author: Jiawei Hu 
 * @Date: 2024-11-29 15:59:27 
 * @Last Modified by:   Jiawei Hu 
 * @Last Modified time: 2024-11-29 15:59:27 
"""
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax

from model.encoder import Encoder
from model.decoder import Decoder

class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    """
    A standard Transformer architecture. Base for this and many
    other models.
    """

    def __init__(self, src_pad_idx, tgt_pad_idx, tgt_bos_idx, enc_voc_size,
                 dec_voc_size, d_model, n_head, max_len, d_ff, n_layers, dropout, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.tgt_bos_idx = tgt_bos_idx
        self.device = device
        self.encoder = Encoder(enc_voc_size=enc_voc_size,
                            max_len=max_len,
                            n_layers=n_layers,
                            n_head=n_head,
                            d_model=d_model,
                            d_ff=d_ff,
                            dropout=dropout)
        self.decoder = Decoder(dec_voc_size=dec_voc_size,
                            max_len=max_len,
                            n_layers=n_layers,
                            n_head=n_head,
                            d_model=d_model,
                            d_ff=d_ff,
                            dropout=dropout)

        self.generator = Generator(d_model, dec_voc_size)

    def forward(self, src, tgt):
        "Take in and process masked src and target sequences."
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_src = self.encoder(src, src_mask)
        dec_tgt = self.decoder(tgt, enc_src, src_mask, tgt_mask)
        return self.generator(dec_tgt)

    def make_src_mask(self, src):
        """
        Mask the padding tokens int source sentence.
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_tgt_mask(self, tgt):
        """
        Mask the padding tokens int target sentence.
        """
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(3)
        tgt_len = tgt.shape[1]
        tgt_sub_mask = torch.tril(torch.ones(tgt_len, tgt_len)).type(torch.ByteTensor).to(self.device)
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask