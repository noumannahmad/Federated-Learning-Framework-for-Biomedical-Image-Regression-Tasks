
import numpy as np
import torch

def normalize(image):
    return (image - np.mean(image)) / (np.std(image) + 1e-8)

def to_tensor(image, label):
    image = normalize(image)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
    label = torch.tensor(label, dtype=torch.float32)
    return image, label
