import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import requests
from tqdm import tqdm

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.nexar_dataset import NexarDataModule
from src.models.yolo.yolo_model import YOLOModel
from src.models.efficientdet.efficientdet_model import EfficientDetModel
from src.models.lstm.lstm_model import LSTMModel

def train_model(model, data_module, model_name: str):
    """Train a single model."""
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = os.path.join('checkpoints', model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=checkpoint_dir,
        filename=f'{model_name}-{{epoch:02d}}-{{val_loss:.2f}}',
        save_top_k=3,
        mode='min'
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )

    # Initialize logger
    logger = TensorBoardLogger('logs', name=model_name)

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=1,  # Train for just 1 epoch
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=10
    )

    # Train the model
    trainer.fit(model, data_module)
    
    return trainer.checkpoint_callback.best_model_path

def download_yolo_weights():
    """Download YOLOv8 weights if they don't exist."""
    weights_path = "yolov8n.pt"
    if not os.path.exists(weights_path):
        print("Downloading YOLOv8 weights...")
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(weights_path, 'wb') as f, tqdm(
            desc=weights_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
        print("Download completed!")

def main():
    # Set random seed for reproducibility
    pl.seed_everything(42)

    # Download YOLOv8 weights if needed
    download_yolo_weights()

    # Initialize data module
    data_dir = os.path.join('src', 'data', 'raw', 'nexar-collision-prediction')
    data_module = NexarDataModule(
        data_dir=data_dir,
        batch_size=32,
        num_workers=4
    )

    try:
        # Train YOLOv8
        print("Training YOLOv8...")
        yolo_model = YOLOModel(model_name="yolov8n.pt")
        yolo_checkpoint = train_model(yolo_model, data_module, "yolo")
        print(f"YOLOv8 training completed. Best checkpoint: {yolo_checkpoint}")

        # Train EfficientDet
        print("\nTraining EfficientDet...")
        efficientdet_model = EfficientDetModel()
        efficientdet_checkpoint = train_model(efficientdet_model, data_module, "efficientdet")
        print(f"EfficientDet training completed. Best checkpoint: {efficientdet_checkpoint}")

        # Train LSTM
        print("\nTraining LSTM...")
        lstm_model = LSTMModel()
        lstm_checkpoint = train_model(lstm_model, data_module, "lstm")
        print(f"LSTM training completed. Best checkpoint: {lstm_checkpoint}")

        print("\nAll models trained successfully!")
        print("\nBest checkpoints:")
        print(f"YOLOv8: {yolo_checkpoint}")
        print(f"EfficientDet: {efficientdet_checkpoint}")
        print(f"LSTM: {lstm_checkpoint}")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 