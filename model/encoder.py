"""
 * @Author: Jiawei Hu 
 * @Date: 2024-11-29 15:59:27 
 * @Last Modified by:   Jiawei Hu 
 * @Last Modified time: 2024-11-29 15:59:27 
 """

from model.utils import *
from model.embedding import Embedding
from model.attention import MultiHeadedAttention


class EncoderLayer(nn.Module):
    "Encoder layer is made up of self-attn and feed forward sublayers"

    def __init__(self, n_head, d_model, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(n_head, d_model, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        # 2 add & norm sublayers one for self-attn and one for feed forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, src, src_mask):
        "Compute self attention, positionwise feed forward network."
        src = self.sublayer[0](src, lambda src: self.self_attn(src, src, src, src_mask))
        return self.sublayer[1](src, self.feed_forward)


class Encoder(nn.Module):
    "This implements encoder block which is a stack of N encoder layers"

    def __init__(self, enc_voc_size, max_len, n_layers, n_head, d_model, d_ff, dropout):
        super().__init__()
        encoder_layer = EncoderLayer(n_head, d_model, d_ff, dropout)
        self.layers = clones(encoder_layer, n_layers)
        self.emb = Embedding(vocab_size=enc_voc_size,
                            d_model=d_model,
                            max_len=max_len,
                            dropout=dropout)

        self.norm = LayerNorm(encoder_layer.d_model)

    def forward(self, src, src_mask):
        "Pass the input (and mask) through each layer in turn."
        src = self.emb(src)    # embedded the input_ids

        for layer in self.layers:
            src = layer(src, src_mask)
        return self.norm(src)