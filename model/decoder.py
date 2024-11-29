 """
 * @Author: Jiawei Hu 
 * @Date: 2024-11-29 15:59:27 
 * @Last Modified by:   Jiawei Hu 
 * @Last Modified time: 2024-11-29 15:59:27 
"""
from model.utils import *
from model.embedding import Embedding
from model.attention import MultiHeadedAttention


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, n_head, d_model, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = MultiHeadedAttention(d_model=d_model, n_head=n_head)
        self.cross_attn = MultiHeadedAttention(d_model=d_model, n_head=n_head)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        # 3 add & norm sublayers one for self-attn, one for cross-attn and one for feed forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)

    def forward(self, dec, enc, src_mask, tgt_mask):
        "Compute self attention, cross attention, positionwise feed forward network.."

        dec = self.sublayer[0](dec, lambda dec: self.self_attn(dec, dec, dec, tgt_mask))
        dec = self.sublayer[1](dec, lambda dec: self.cross_attn(dec, enc, enc, src_mask))
        return self.sublayer[2](dec, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, dec_voc_size, max_len, n_layers, n_head, d_model, d_ff, dropout):
        super(Decoder, self).__init__()
        decoder_layer = DecoderLayer(n_head, d_model, d_ff, dropout)
        self.layers = clones(decoder_layer, n_layers)
        self.emb = Embedding(vocab_size=dec_voc_size,
                            d_model=d_model,
                            max_len=max_len,
                            dropout=dropout)

        self.norm = LayerNorm(decoder_layer.d_model)

    def forward(self, tgt, enc_src, src_mask, tgt_mask):
        tgt = self.emb(tgt)    # embedded the input_ids

        for layer in self.layers:
            tgt = layer(tgt, enc_src, src_mask, tgt_mask)
        return self.norm(tgt)