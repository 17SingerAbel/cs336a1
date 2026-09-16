import torch
import torch.nn as nn
from einops import rearrange, einsum
from cs336_basics.Linear import Linear
from cs336_basics.RoPE import RoPE
import math


class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self, d_model, num_heads, theta=None, max_seq_len=None, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)


        self.theta = theta
        self.max_seq_len = max_seq_len
    
        
    def forward(self, x, token_positions=None):

        Q = self.q_proj.forward(x)
        Q = rearrange(Q, "... seq (h d) -> ... h seq d", h=self.num_heads)

        K = self.k_proj.forward(x)
        K = rearrange(K, "... seq (h d) -> ... h seq d", h=self.num_heads)

        V = self.v_proj.forward(x)
        V = rearrange(V, "... seq (h d) -> ... h seq d", h=self.num_heads)

        # # apply RoPe
        if self.theta is not None and token_positions is not None:
            rope = RoPE(self.theta, self.d_head, self.max_seq_len, device=x.device, dtype=x.dtype)
            Q = rope.forward(Q, token_positions)
            K = rope.forward(K, token_positions)

        multiHeadScores = einsum(Q, K, '... h queries d_out, ... h keys d_out -> ... h queries keys') / (self.d_head ** 0.5)

        seq_len = x.shape[-2]

        mask = torch.ones(
            seq_len,
            seq_len,
            dtype=torch.bool,
            device=x.device
        )
        mask = torch.triu(mask, diagonal=1)

        multiHeadScores = multiHeadScores.masked_fill(mask, -torch.inf)

        max_entity = torch.amax(multiHeadScores, dim=-1, keepdim=True)
        exp_scores = torch.exp(multiHeadScores - max_entity)
        multiHeadScores = exp_scores / torch.sum(exp_scores, dim=-1, keepdim=True)

         
        attentions = einsum(multiHeadScores, V, '... h queries keys, ... h keys values -> ... h queries values')

        attentions = rearrange(attentions, '... h seq values -> ... seq (h values)')


        return self.output_proj.forward(attentions)