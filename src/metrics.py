import torch

def mae(pred, target):
    """
    Mean Absolute Error
    """
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target)
    return torch.mean(torch.abs(pred - target)).item()


def r2_score(pred, target):
    """
    R² Score (Coefficient of Determination)
    """
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target)

    mean_target = torch.mean(target)
    ss_tot = torch.sum((target - mean_target) ** 2)
    ss_res = torch.sum((target - pred) ** 2)
    
    # Avoid division by zero
    if ss_tot == 0:
        return float("nan")
    
    return (1 - ss_res / ss_tot).item()
