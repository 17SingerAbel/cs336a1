import torch
import torch.nn as nn
from torch import Tensor
from einops import  einsum, reduce

class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.weights = nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype,
            ),
            requires_grad=True
        )

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # x: batch seq d_model
        # → element-wise square
        # → mean along the last d_model dimension
        # → add epsilon
        # → square root
        # → x divided by RMS
        # → element-wise multiply by g
        rms_sum = reduce(x**2, "... d_model -> ... 1", "mean")  
        normalized_x = x /(rms_sum + self.eps) ** 0.5

        result = einsum(normalized_x, self.weights, '... d, d -> ... d')


        return result.to(in_dtype)