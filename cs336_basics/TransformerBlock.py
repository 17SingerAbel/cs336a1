import torch
import torch.nn as nn
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.MultiHeadSelfAttention import MultiHeadSelfAttention
from cs336_basics.SwiGLU import SwiGlu

class TransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, device=None, dtype=None):
        super().__init__()

        
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model, 
            num_heads, 
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype
        )

        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGlu(d_model, d_ff, device=device, dtype=dtype )

    def forward(self, x, seq_len):
        token_positions = torch.arange(seq_len, device=x.device)
        normalized_x = self.ln1.forward(x)
        residual_attention = x + self.attn.forward(normalized_x, token_positions)

        normalized_attention = self.ln2.forward(residual_attention)
        ffn_output = self.ffn.forward(normalized_attention)

        return residual_attention + ffn_output
        


