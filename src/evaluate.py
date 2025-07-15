import torch
from src.metrics import mae, r2_score
from utils.logger import setup_logger

def evaluate(model, dataloader, criterion, device, log_file=None):
    """
    Evaluate the model on the given dataloader.

    Args:
        model: Trained PyTorch model
        dataloader: Validation or test DataLoader
        criterion: Loss function (e.g., MSELoss)
        device: 'cuda' or 'cpu'
        log_file: Optional path to log results

    Returns:
        avg_loss: Mean loss across dataset
        metrics: Dictionary with MAE and R² score
    """
    logger = setup_logger("eval_logger", log_file) if log_file else None
    model.eval()
    total_loss = 0.0
    preds, targets = [], []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data).squeeze()

            loss = criterion(output, target.float())
            total_loss += loss.item()

            preds.append(output.cpu())
            targets.append(target.cpu())

    # Concatenate results
    preds = torch.cat(preds)
    targets = torch.cat(targets)

    # Compute additional metrics
    eval_mae = mae(preds, targets)
    eval_r2 = r2_score(preds, targets)
    avg_loss = total_loss / len(dataloader)

    # Log
    if logger:
        logger.info(f"Avg Loss: {avg_loss:.4f} | MAE: {eval_mae:.4f} | R²: {eval_r2:.4f}")
    else:
        print(f"🧪 Evaluation - Loss: {avg_loss:.4f}, MAE: {eval_mae:.4f}, R²: {eval_r2:.4f}")

    return avg_loss, {"mae": eval_mae, "r2": eval_r2}
