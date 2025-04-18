import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from dataloader import get_dataloader
from model import get_efficientdet_model
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_one_epoch(model, dataloader, optimizer, device, scaler):
    model.train()
    total_loss = 0
    for batch_idx, (images, targets) in enumerate(dataloader):
        if not images.size(0):  # Skip empty batches
            continue
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with autocast():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += losses.item()
        if batch_idx % 10 == 0:
            logger.info(f"Batch {batch_idx}, Loss: {losses.item():.4f}")
    return total_loss / max(1, len(dataloader))

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets in dataloader:
            if not images.size(0):
                continue
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
    return total_loss / max(1, len(dataloader))

def main():
    # Hyperparameters
    num_classes = 5  # 4 classes + background
    batch_size = 16
    epochs = 20
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Dataset
    data_dir = './dataset'  # Path to unzipped dataset
    train_loader = get_dataloader(data_dir, split='train', batch_size=batch_size)
    val_loader = get_dataloader(data_dir, split='valid', batch_size=batch_size)

    # Model
    model = get_efficientdet_model(num_classes=num_classes)
    model.to(device)

    # Optimizer and scaler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        val_loss = validate(model, val_loader, device)
        logger.info(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'efficientdet_d0_best.pth'))
            logger.info(f"Saved best model at epoch {epoch+1}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'efficientdet_d0_final.pth'))
    logger.info("Training completed")

if __name__ == '__main__':
    main()