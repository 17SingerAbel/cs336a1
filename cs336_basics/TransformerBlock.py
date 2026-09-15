import torch
import torch.nn as nn
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.MultiHeadSelfAttention import MultiHeadSelfAttention
from cs336_basics.SwiGLU import SwiGlu

class TransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, weights=None, device=None, dtype=None):
        super().__init__()

        rmsNorm1_weights = weights['ln1.weight'] if weights is not None else None
        self.rmsNorm_1 = RMSNorm(d_model, weights=rmsNorm1_weights, device=device, dtype=dtype)


        if weights is not None:
            transformer_weights = weights['attn']
        else:
            transformer_weights = None
            
        self.multihead = MultiHeadSelfAttention(
            d_model, num_heads, 
            weights=transformer_weights
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype
        )

        rmsNorm2_weights = weights['ln2.weight'] if weights is not None else None
        self.rmsNorm_2 = RMSNorm(d_model, weights=rmsNorm2_weights, device=device, dtype=dtype)

        swiglu_weights = weights['ffn'] if weights is not None else None
        self.swiglu = SwiGlu(d_model, d_ff, weights=swiglu_weights, device=device, dtype=dtype )

    def forward(self, x, seq_len):
        token_positions = torch.arange(seq_len, device=x.device)
        normalized_x = self.rmsNorm_1.forward(x)
        residual_attention = x + self.multihead.forward(normalized_x, token_positions)

        normalized_attention = self.rmsNorm_2.forward(residual_attention)
        ffn_output = self.swiglu.forward(normalized_attention)

        return residual_attention + ffn_output
        


