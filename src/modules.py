import torch
from torch import nn


class SvegaLinear(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super().__init__()

        self.weights = nn.Parameter(torch.empty(input_dimension, output_dimension))
        self.biases = nn.Parameter(torch.empty(output_dimension))

        nn.init.kaiming_uniform_(self.weights)
        nn.init.zeros_(self.biases)
        
    def forward(self, X):
        return X @ self.weights + self.biases
  

class SvegaLeakyReLU(nn.Module):
    def __init__(self, alpha=0):
        super().__init__()
        self.alpha = alpha

    def forward(self, X):
        return torch.where(X > 0, X, X * self.alpha)
    

class SvegaFlatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return torch.flatten(X, start_dim=1)
    

class SvegaLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-05):
        super().__init__()

        self.eps = eps

        self.scale_parameters = nn.Parameter(torch.ones(hidden_size))
        self.shift_parameters = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, X):
        means = torch.mean(X, dim=-1, keepdim=True)
        variances = torch.sqrt(X.var(dim=-1, keepdim=True, unbiased=False) + self.eps)

        normalized_X = (X - means) / variances
        scaled_X = normalized_X * self.scale_parameters
        shifted_x = scaled_X + self.shift_parameters
        
        return shifted_x
