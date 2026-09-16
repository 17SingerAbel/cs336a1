import torch
from torch import nn
import math
from einops import einsum
from cs336_basics.Linear import Linear

class SwiGlu(nn.Module):

    def __init__(self, d_model, d_ff, device=None, dtype=None ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
            

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = self.w1.forward(x)
        silu_output = torch.sigmoid(w1_x) * w1_x

        w3_x = self.w3.forward(x)
        hidden = silu_output * w3_x

        return self.w2.forward(hidden)

    