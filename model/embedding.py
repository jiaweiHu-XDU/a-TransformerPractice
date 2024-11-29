"""
@author: Zhihao Li
@date: 2024-11-11
@homepage: https://zhihaoli.top/
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    # Implement the position encoding (PE) function.

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # adds token embedding to its position embedding
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """

    def __init__(self, vocab_size, d_model):
        """
        class for token embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


class Embedding(nn.Module):
    
    def __init__(self, vocab_size, d_model, max_len, dropout=0.1):
        super(Embedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, dropout, max_len)
        
    def forward(self, x):
        """
        input_ids for tokens.
        """
        return self.pos_emb(self.tok_emb(x))
    

