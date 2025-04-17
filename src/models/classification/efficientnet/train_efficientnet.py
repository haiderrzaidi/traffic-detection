import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
os.chdir(project_root)  # Change working directory to project root

from data.nexar_dataset import NexarDataModule
from models.classification.efficientnet.efficientnet_model import EfficientNetModel
from configs.model_config import EFFICIENTNET_CONFIG, DATASET_CONFIG

def train_efficientnet():
    """Train the EfficientNet model."""
    print("\nTraining EfficientNet model...")
    
    # Set random seed for reproducibility
    pl.seed_everything(42)
    
    # Initialize data module with absolute path
    data_dir = os.path.join(project_root, 'data', 'raw', 'nexar-collision-prediction')
    print(f"Using data directory: {data_dir}")
    
    data_module = NexarDataModule(
        data_dir=data_dir,
        batch_size=DATASET_CONFIG['batch_size'],
        num_workers=DATASET_CONFIG['num_workers']
    )
    
    # Initialize model
    model = EfficientNetModel(
        model_name=EFFICIENTNET_CONFIG['model_name'],
        pretrained=EFFICIENTNET_CONFIG['pretrained'],
        num_classes=EFFICIENTNET_CONFIG['num_classes'],
        learning_rate=1e-4,
        weight_decay=1e-5
    )
    
    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(project_root, 'checkpoints', 'efficientnet'),
        filename='efficientnet-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
        save_last=True  # Save the last checkpoint
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )
    
    # Initialize logger
    logger = TensorBoardLogger(os.path.join(project_root, 'logs'), name='efficientnet')
    
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
    print("Starting training...")
    trainer.fit(model, data_module)
    
    # Test the model
    print("Starting testing...")
    trainer.test(model, data_module)
    
    # Save the final model
    final_checkpoint_path = os.path.join(project_root, 'checkpoints', 'efficientnet', 'last.ckpt')
    trainer.save_checkpoint(final_checkpoint_path)
    print(f"Saved final model checkpoint to: {final_checkpoint_path}")

if __name__ == '__main__':
    train_efficientnet() 