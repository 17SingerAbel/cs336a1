import torch
import torch.nn as nn
from einops import rearrange, einsum
from cs336_basics.RoPE import RoPE
import math

class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self, d_model, num_heads, weights=None, theta=None, max_seq_len=None, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        if weights is None:
            self.q_proj = nn.Parameter(weights['q_proj.weight'], requires_grad = True)
            self.k_proj = nn.Parameter(weights['v.weight'], requires_grad = True)
            self.v_proj = nn.Parameter(weights['k_proj.weight'], requires_grad = True)
            self.w_o = nn.Parameter(weights['output_proj.weight'], requires_grad = True)
        else:   
            self.q_proj = nn.Parameter(
                            torch.empty(d_model, d_model, device=device, dtype=dtype),
                            requires_grad=True
                        )

            self.k_proj = nn.Parameter(
                            torch.empty(d_model, d_model, device=device, dtype=dtype),
                            requires_grad=True
                        )

            self.v_proj = nn.Parameter(
                            torch.empty(d_model, d_model, device=device, dtype=dtype),
                            requires_grad=True
                        )

            self.w_o = nn.Parameter(
                            torch.empty(d_model, d_model, device=device, dtype=dtype),
                            requires_grad=True
                        )

            std = math.sqrt(2 / (d_model + d_model))
            nn.init.trunc_normal_(
                self.q_proj,
                mean=0.0,
                std=std,
                a=-3*std,
                b=3*std,
            )
            nn.init.trunc_normal_(
                            self.k_proj,
                            mean=0.0,
                            std=std,
                            a=-3*std,
                            b=3*std,
                        )
            nn.init.trunc_normal_(
                            self.v_proj,
                            mean=0.0,
                            std=std,
                            a=-3*std,
                            b=3*std,
                        )

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


        return einsum(attentions, self.w_o, '... seq d_in, d_out d_in -> ... seq d_out')
        