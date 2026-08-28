import math

import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )

        std = math.sqrt(2 / (in_features + out_features))

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return  einsum(x, self.weight, "... in_feature, out_feature in_feature -> ... out_feature")

# linear = Linear(3, 2)

# x = torch.randn(4, 5, 3)
# y = linear(x)

# print(linear.weight.shape)  # 预期？ 2,3
# print(y.shape)              # 预期？ 4, 5, 3 @ W.T, 3. 2