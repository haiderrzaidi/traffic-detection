import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.nexar_dataset import NexarDataModule
from models.resnet import ResNetModel
from models.efficientnet import EfficientNetModel
from models.vit import ViTModel
from models.lstm import LSTMModel
from models.gru import GRUModel
from configs.model_config import (
    RESNET_CONFIG,
    EFFICIENTNET_CONFIG,
    VIT_CONFIG,
    LSTM_CONFIG,
    GRU_CONFIG,
    DATASET_CONFIG
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
    elif model_name == 'efficientnet':
        model = EfficientNetModel(
            model_name=EFFICIENTNET_CONFIG['model_name'],
            pretrained=EFFICIENTNET_CONFIG['pretrained'],
            num_classes=EFFICIENTNET_CONFIG['num_classes'],
            learning_rate=1e-4,
            weight_decay=1e-5
        )
    elif model_name == 'vit':
        model = ViTModel(
            model_name=VIT_CONFIG['model_name'],
            pretrained=VIT_CONFIG['pretrained'],
            num_classes=VIT_CONFIG['num_classes'],
            learning_rate=1e-4,
            weight_decay=1e-5
        )
    elif model_name == 'lstm':
        model = LSTMModel(
            input_size=LSTM_CONFIG['input_size'],
            hidden_size=LSTM_CONFIG['hidden_size'],
            num_layers=LSTM_CONFIG['num_layers'],
            dropout=LSTM_CONFIG['dropout'],
            bidirectional=LSTM_CONFIG['bidirectional'],
            num_classes=2,
            learning_rate=1e-4,
            weight_decay=1e-5
        )
    elif model_name == 'gru':
        model = GRUModel(
            input_size=GRU_CONFIG['input_size'],
            hidden_size=GRU_CONFIG['hidden_size'],
            num_layers=GRU_CONFIG['num_layers'],
            dropout=GRU_CONFIG['dropout'],
            bidirectional=GRU_CONFIG['bidirectional'],
            num_classes=2,
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

def main():
    # Set random seed for reproducibility
    pl.seed_everything(42)

    # Initialize data module
    data_dir = os.path.join('src', 'data', 'raw', 'nexar-collision-prediction')
    print(f"Using data directory: {data_dir}")
    
    data_module = NexarDataModule(
        data_dir=data_dir,
        batch_size=DATASET_CONFIG['batch_size'],
        num_workers=DATASET_CONFIG['num_workers']
    )

    # Train all models
    models_to_train = ['resnet', 'efficientnet', 'vit', 'lstm', 'gru']
    for model_name in models_to_train:
        try:
            train_model(model_name, data_module)
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue

if __name__ == '__main__':
    main() 