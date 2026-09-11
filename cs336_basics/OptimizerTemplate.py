import torch
import torch.nn as nn
from collections.abc import Callable, Iterable
from typing import Optional

class Optimizer(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3):
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                p.data -= lr / (t+1)**0.5 * p.grad.data
                state["t"] = t + 1
        return loss

# for lr in [1e1, 1e2, 1e3]:
#     print("lr: ", lr)
#     weights = nn.Parameter(5 * torch.randn((10, 10)))
#     opt = Optimizer([weights], lr=lr)

#     for t in range(10):
#         opt.zero_grad()
#         loss = (weights **2).mean()
#         print(loss.cpu().item())
#         loss.backward()
#         opt.step()