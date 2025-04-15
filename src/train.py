import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from data.nexar_dataset import NexarDataModule
from models.resnet import ResNetModel
from configs.model_config import RESNET_CONFIG, DATASET_CONFIG

def main():
    # Set random seed for reproducibility
    pl.seed_everything(42)

    # Initialize data module
    data_dir = os.path.join('src', 'data', 'raw', 'nexar-collision-prediction')
    data_module = NexarDataModule(
        data_dir=data_dir,
        batch_size=32,
        num_workers=4
    )

    # Initialize model
    model = ResNetModel(
        num_classes=RESNET_CONFIG['num_classes'],
        pretrained=RESNET_CONFIG['pretrained'],
        learning_rate=1e-4,
        weight_decay=1e-5
    )

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='resnet-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )

    # Initialize logger
    logger = TensorBoardLogger('logs', name='resnet')

    # Initialize trainer with just 1 epoch
    trainer = pl.Trainer(
        max_epochs=1,  # Run for just 1 epoch
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=10
    )

    # Train the model
    trainer.fit(model, data_module)

    # Test the model
    trainer.test(model, data_module)

if __name__ == '__main__':
    main() 