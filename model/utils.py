import copy
import torch
import torch.nn.functional as F
import torch.nn as nn

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layer norm module "

    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class FeedForward(nn.Module):
    
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        return self.linear_2(x)


class SublayerConnection(nn.Module):
    """
    sublayerconnection is a building block of an encoder layer and a decoder layer.
    The encoder layer has two sublayers: self-attn, and feedforwrad.
    The decoder has three sublayers: self-attn, cross-attn, and feedforward.
    Each sublayer applies a layer normalization and a residual connection.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        Input sublayer can be self-attn or feedforward.
        """
        return x + self.dropout(sublayer(self.norm(x)))