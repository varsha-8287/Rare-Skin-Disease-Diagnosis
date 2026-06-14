import torch

def poincare_distance(x, y, eps=1e-5):
    x_norm = torch.clamp(torch.norm(x, dim=-1, keepdim=True), max=1 - eps)
    y_norm = torch.clamp(torch.norm(y, dim=-1, keepdim=True), max=1 - eps)

    diff = x.unsqueeze(1) - y.unsqueeze(0)
    dist_sq = torch.sum(diff ** 2, dim=-1)

    denom = (1 - x_norm ** 2) * (1 - y_norm.transpose(0, 1) ** 2)
    cosh_arg = 1 + 2 * dist_sq / denom.clamp(min=eps)
    return torch.acosh(cosh_arg.clamp(min=1 + eps))

def expmap0(v, eps=1e-5):
    """Map Euclidean vectors to the Poincaré ball from the origin.
    This ensures all features live properly in hyperbolic space."""
    norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=eps)
    return torch.tanh(norm) * v / norm