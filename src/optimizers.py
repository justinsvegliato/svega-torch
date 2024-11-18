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


class SvegaAdaGrad(Optimizer):
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

                if "grad_squared" not in self.state[param]:
                    self.state[param]["grad_squared"] = torch.zeros_like(param)

                grad_squared = self.state[param]["grad_squared"]
                grad_squared.add_(grad.pow(2))

                std = grad_squared.sqrt().add(1e-7)
                update = param_group["lr"] * grad / std

                param.data.add_(-update)

        return loss
    

class SvegaRMSProp(Optimizer):
    def __init__(self, params, lr=0.01, decay_rate=0.9):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if decay_rate < 0.0:
            raise ValueError(f"Invalid decay rate: {decay_rate}")
        
        defaults = {"lr": lr, "decay_rate": decay_rate}
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

                if "grad_squared" not in self.state[param]:
                    self.state[param]["grad_squared"] = torch.zeros_like(param)

                grad_squared = self.state[param]["grad_squared"]
                grad_squared.mul_(param_group["decay_rate"]).addcmul_(grad, grad, value=(1 - param_group["decay_rate"]))

                std = grad_squared.sqrt().add(1e-7)
                update = param_group["lr"] * grad / std

                param.data.add_(-update)

        return loss
    

class SvegaAdam(Optimizer):
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if beta1 < 0.0:
            raise ValueError(f"Invalid beta1: {beta1}")
        
        if beta2 < 0.0:
            raise ValueError(f"Invalid beta2: {beta2}")
    
        self.time = 0
        
        defaults = {"lr": lr, "beta1": beta1, "beta2": beta2}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.time += 1

        for param_group in self.param_groups:
            for param in param_group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data

                if "first_moment" not in self.state[param]:
                    self.state[param]["first_moment"] = torch.zeros_like(param)

                if "second_moment" not in self.state[param]:
                    self.state[param]["second_moment"] = torch.zeros_like(param)

                first_moment = self.state[param]["first_moment"]
                first_moment.mul_(param_group["beta1"]).add_(grad, alpha=(1 - param_group["beta1"]))
                bias_corrected_first_moment = first_moment / (1 - param_group["beta1"] ** self.time)

                second_moment = self.state[param]["second_moment"]
                second_moment.mul_(param_group["beta2"]).addcmul_(grad, grad, value=(1 - param_group["beta2"]))
                bias_corrected_second_moment = second_moment / (1 - param_group["beta2"] ** self.time)

                std = bias_corrected_second_moment.sqrt().add(1e-7)
                update = param_group["lr"] * (bias_corrected_first_moment / std)

                param.data.add_(-update)

        return loss
