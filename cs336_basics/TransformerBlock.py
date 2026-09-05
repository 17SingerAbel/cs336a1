import torch
import torch.nn as nn
import MultiHeadSelfAttention
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.MultiHeadSelfAttention import MultiHeadSelfAttention

# class TransformerBlock(nn.Module):

#     def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, weights, seq_len):
#         super().__init__()
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_ff = d_ff
#         self.max_seq_len = max_seq_len
#         self.theta = theta
#         self.q_proj_weight = weights.attn.q_proj.weight
#         self.k_proj_weight = weights.attn.k_proj.weight
#         self.v_proj_weight = weights.attn.v_proj.weight
#         self.o_proj_weight = weights.attn.output_proj.weight
#         self.rmsNorm_1 = RMSNorm(d_model)
#         self.rmsNorm_1.weights = weights.ln1.weight
#         token_positions = torch.arange(seq_len)
#         self.multihead = MultiHeadSelfAttention(d_model, num_heads, self.q_proj_weight, self.k_proj_weight, self.v_proj_weight, self.o_proj_weight, theta=theta, token_positions=token_positions, max_seq_len=max_seq_len)
#         self.rmsNorm_2 = RMSNorm(d_model)
#         self.rmsNorm_2.weights = weights.ln2.weight
#         self.ffn_w1  = weights.ffn.w1.weight
#         self.ffn_ww  = weights.ffn.ww.weight
#         self.ffn_w3  = weights.ffn.w3.weight

#     def forward(self, x):
#         normalized_x = self.rmsNorm.forward(x)
#         residual_attention = x + self.multihead.forward(normalized_x)
#         # ？？ dff is not used
#         normalized_attention = self.rmsNorm.forward(residual_attention)
#         
#
#
