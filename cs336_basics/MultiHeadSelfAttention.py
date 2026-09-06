import torch
import torch.nn as nn
from einops import rearrange, einsum
from cs336_basics.RoPE import RoPE


class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self, d_model, num_heads,q_proj, k_proj, v_proj, w_o, theta=None, max_seq_len=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = nn.Parameter(q_proj, requires_grad = True)
        self.k_proj = nn.Parameter(k_proj, requires_grad = True)
        self.v_proj = nn.Parameter(v_proj, requires_grad = True)
        self.w_o = nn.Parameter(w_o, requires_grad = True)
        self.theta = theta
        self.max_seq_len = max_seq_len
    
        
    def forward(self, x, token_positions):
        
    
        Q = einsum(x, self.q_proj, '... seq d_in, d_out d_in -> ... seq d_out')
        Q = rearrange(Q, "... seq (h d) -> ... h seq d", h=self.num_heads)

        K = einsum(x, self.k_proj, '... seq d_in, d_out d_in -> ... seq d_out')
        K = rearrange(K, "... seq (h d) -> ... h seq d", h=self.num_heads)


        V = einsum(x, self.v_proj, '... seq d_in, d_out d_in -> ... seq d_out')
        V = rearrange(V, "... seq (h d) -> ... h seq d", h=self.num_heads)

        # # apply RoPe
        if self.theta is not None:
            rope = RoPE(self.theta, self.d_head, self.max_seq_len)
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


        return einsum(attentions, self.w_o, '... seq d_in, d_out d_in -> ... seq d_out')
        