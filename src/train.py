import os
import torch
from utils.logger import setup_logger

def train(model, dataloader, optimizer, criterion, device, epoch=1, scheduler=None, save_path=None, log_file=None):
    """
    Train the model for one epoch.

    Args:
        model: PyTorch model to be trained
        dataloader: Training DataLoader
        optimizer: Optimizer (e.g., Adam, SGD)
        criterion: Loss function (e.g., MSELoss)
        device: 'cuda' or 'cpu'
        epoch: Current epoch number
        scheduler: Optional learning rate scheduler
        save_path: Optional path to save model checkpoint
        log_file: Optional path for log output

    Returns:
        avg_loss: Average loss over all batches
    """
    logger = setup_logger("train_logger", log_file) if log_file else None
    model.train()
    total_loss = 0.0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.squeeze(), target.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if logger:
            logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")
        else:
            if batch_idx % 10 == 0:
                print(f"[Epoch {epoch}] Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)

    if scheduler:
        scheduler.step()

    print(f"[Epoch {epoch}] Average Loss: {avg_loss:.4f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"✅ Model saved to {save_path}")

    return avg_loss
