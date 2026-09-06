import torch
import torch.nn as nn
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.MultiHeadSelfAttention import MultiHeadSelfAttention
from cs336_basics.SwiGLU import SwiGlu

class TransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, weights, seq_len):
        super().__init__()

        self.rmsNorm_1 = RMSNorm(d_model, weights=weights['ln1.weight'])
        token_positions = torch.arange(seq_len)

        self.multihead = MultiHeadSelfAttention(
            d_model, num_heads, 
            weights['attn.q_proj.weight'], 
            weights['attn.k_proj.weight'], 
            weights['attn.v_proj.weight'], 
            weights['attn.output_proj.weight'], 
            theta=theta, token_positions=token_positions, 
            max_seq_len=max_seq_len)
        
        self.rmsNorm_2 = RMSNorm(d_model, weights=weights['ln2.weight'])
        self.swiglu = SwiGlu(d_model, d_ff)
        self.swiglu.w1_weight = nn.Parameter(weights['ffn.w1.weight'], requires_grad=True)
        self.swiglu.w2_weight = nn.Parameter(weights['ffn.w2.weight'], requires_grad=True)
        self.swiglu.w3_weight = nn.Parameter(weights['ffn.w3.weight'], requires_grad=True)

    def forward(self, x):
        normalized_x = self.rmsNorm_1.forward(x)
        residual_attention = x + self.multihead.forward(normalized_x)

        normalized_attention = self.rmsNorm_2.forward(residual_attention)
        ffn_output = self.swiglu.forward(normalized_attention)
        return residual_attention + ffn_output
        


