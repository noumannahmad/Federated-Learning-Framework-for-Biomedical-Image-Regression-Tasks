
import torch

def add_gaussian_noise(tensor, mean=0.0, std=1e-3):
    noise = torch.randn_like(tensor) * std + mean
    return tensor + noise

def clip_gradients(model, clip_value=1.0):
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data = torch.clamp(param.grad.data, -clip_value, clip_value)
