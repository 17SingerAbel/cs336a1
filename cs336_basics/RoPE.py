import torch
from torch import nn
from einops import einsum


class RoPE(nn.Module):
    def __init__(self, theta, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))

        # idx = torch.arange(start=1, end=k, step=1)
        # theta_k = 1  / theta ** ((2*idx - 2) / d_k)
        self.register_buffer("freq", freq)
        positions = torch.arange(max_seq_len, device=device)

        angles = einsum(positions, freq, 'row, column -> row column')
        self.register_buffer('cos_cached', torch.cos(angles))
        self.register_buffer('sin_cached', torch.sin(angles))
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:

        cos = self.cos_cached[token_positions]  # [..., sequence, d_k/2]
        sin = self.sin_cached[token_positions]  # [..., sequence, d_k/2]

        x_even = x[..., 0::2]  #[..., seq, d_k/2]
        x_odd = x[..., 1::2]


        y_even = x_even * cos - x_odd * sin
        y_odd = x_even * sin + x_odd * cos

        return torch.stack([y_even, y_odd], dim=-1).reshape_as(x)