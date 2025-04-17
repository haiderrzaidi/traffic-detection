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

from data.nexar_dataset import NexarDataModule
from models.resnet import ResNetModel
from models.lstm.lstm_model import LSTMModel
from models.yolo.yolo_model import YOLOModel
from models.faster_rcnn import FasterRCNNModel
from configs.model_config import (
    RESNET_CONFIG,
    LSTM_CONFIG,
    YOLO_CONFIG,
    DATASET_CONFIG,
    FASTER_RCNN_CONFIG
)

def train_model(model_name: str, data_module: NexarDataModule):
    """Train a specific model."""
    print(f"\nTraining {model_name} model...")
    
    # Initialize model based on name
    if model_name == 'resnet':
        model = ResNetModel(
            num_classes=RESNET_CONFIG['num_classes'],
            pretrained=RESNET_CONFIG['pretrained'],
            learning_rate=1e-4,
            weight_decay=1e-5
        )
    elif model_name == 'lstm':
        model = LSTMModel(
            input_size=LSTM_CONFIG['input_size'],
            hidden_size=LSTM_CONFIG['hidden_size'],
            num_layers=LSTM_CONFIG['num_layers'],
            dropout=LSTM_CONFIG['dropout'],
            num_classes=2,
            learning_rate=1e-4,
        )
    elif model_name == 'yolo':
        model = YOLOModel(
            model_name=YOLO_CONFIG['model_name'],
            pretrained=YOLO_CONFIG['pretrained'],
            learning_rate=1e-4,
            confidence_threshold=0.5
        )
    elif model_name == 'fasterrcnn':
        model = FasterRCNNModel(
            num_classes=FASTER_RCNN_CONFIG['num_classes'],
            backbone=FASTER_RCNN_CONFIG['backbone'],
            pretrained_backbone=FASTER_RCNN_CONFIG['pretrained_backbone'],
            min_size=FASTER_RCNN_CONFIG['min_size'],
            max_size=FASTER_RCNN_CONFIG['max_size'],
            learning_rate=1e-4,
            weight_decay=1e-5
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
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
        max_epochs=1,  # Run for just 1 epoch initially
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=10,
        precision=16 if torch.cuda.is_available() else 32  # Use mixed precision if GPU available
    )

    # Train the model
    print(f"Starting training for {model_name}...")
    trainer.fit(model, data_module)

    # Test the model
    print(f"Starting testing for {model_name}...")
    trainer.test(model, data_module)

    # Save the final model weights
    weights_dir = os.path.join('checkpoints', 'final_weights')
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, f'{model_name}_final_weights.pth')
    torch.save(model.state_dict(), weights_path)
    print(f"Final weights saved to: {weights_path}")

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
    print(f"Using data directory: {data_dir}")
    
    data_module = NexarDataModule(
        data_dir=data_dir,
        batch_size=DATASET_CONFIG['batch_size'],
        num_workers=DATASET_CONFIG['num_workers']
    )

    # Train only YOLO, ResNet, and LSTM models
    models_to_train = ["resnet",'lstm']
    for model_name in models_to_train:
        try:
            print(f"\n{'='*50}")
            print(f"Starting training for {model_name} model")
            print(f"{'='*50}")
            train_model(model_name, data_module)
            print(f"\n{'='*50}")
            print(f"Completed training for {model_name} model")
            print(f"{'='*50}\n")
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            print(f"Skipping {model_name} and continuing with next model...")
            continue

if __name__ == '__main__':
    main() 