import torch
from torch.optim import Optimizer


class SvegaSGD(Optimizer):
    def __init__(self, params, lr=0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for param_group in self.param_groups:
            for param in param_group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.data
                param.data = param.data - param_group["lr"] * grad

        return loss
