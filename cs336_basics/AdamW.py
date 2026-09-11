import torch
import torch.nn as nn
from collections.abc import Callable, Iterable
from typing import Optional

class AdamW(torch.optim.Optimizer):

    def __init__(self, params, betas, weight_decay, lr=1e-3, eps=1e-8):

        defaults = {"lr": lr}
        super().__init__(params, defaults)
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.weight_decay = weight_decay
        self.eps = eps


    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    t = 1
                    m = torch.zeros_like(p.data)
                    v = torch.zeros_like(p.data)
                else:
                    t = state.get('t')
                    m = state.get('m')
                    v = state.get('v')

                gradient = p.grad.data
                lr_t = lr * (1 - (self.beta2 ** t)) ** 0.5 / (1 - self.beta1 ** t)

                p.data -= lr * self.weight_decay * p.data

                new_m = self.beta1 * m + (1 - self.beta1) * gradient
                new_v = self.beta2 * v + (1 - self.beta2) * (gradient ** 2)
            

                p.data -= lr_t * new_m / (new_v ** 0.5 + self.eps)

                state["t"] = t + 1
                state['m'] = new_m
                state['v'] = new_v
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