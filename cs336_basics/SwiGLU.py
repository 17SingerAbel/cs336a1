import torch
from torch import nn
import math
from einops import einsum

class SwiGlu(nn.Module):

    def __init__(self, d_model, d_ff, weights=None, device=None, dtype=None ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff


        if weights is None:
            self.w1_weight =  nn.Parameter(
                                    torch.empty(
                                        d_ff,
                                        d_model,
                                        device=device,
                                        dtype=dtype,
                                    )
                                )
            self.w2_weight =  nn.Parameter(
                                    torch.empty(
                                        d_model,
                                        d_ff,
                                        device=device,
                                        dtype=dtype,
                                    )
                                )
            self.w3_weight =  nn.Parameter(
                                    torch.empty(
                                        d_ff,
                                        d_model,
                                        device=device,
                                        dtype=dtype,
                                    )
                                )
            
            std = math.sqrt(2 / (d_model + d_ff))

            for weight in (self.w1_weight, self.w2_weight, self.w3_weight):
                nn.init.trunc_normal_(
                    weight,
                    mean=0.0,
                    std=std,
                    a=-3 * std,
                    b=3 * std,
                )
        else:
            self.w1_weight = nn.Parameter(weights['w1.weight'], requires_grad=True)
            self.w2_weight = nn.Parameter(weights['w2.weight'], requires_grad=True)
            self.w3_weight = nn.Parameter(weights['w3.weight'], requires_grad=True)
            

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = einsum(x, self.w1_weight, '... d_model, d_ff d_model -> ... d_ff')
        silu_output = torch.sigmoid(w1_x) * w1_x

        w3_x = einsum(x, self.w3_weight, '... d_model, d_ff d_model -> ... d_ff')

        hidden = silu_output * w3_x

        return einsum(hidden, self.w2_weight, '... d_ff, d_model d_ff -> ... d_model')

        # return  einsum(x, self.weight, "... in_feature, out_feature in_feature -> ... out_feature")
    