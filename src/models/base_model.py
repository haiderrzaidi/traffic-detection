from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim

class BaseModel(pl.LightningModule):
    """Base model class for all models in the project."""
    
    def __init__(self, learning_rate: float = 1e-4):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    @abstractmethod
    def forward(self, x):
        """Forward pass of the model."""
        pass

    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        self.log('test_loss', loss)
        return loss

    @abstractmethod
    def compute_loss(self, y_hat, y):
        """Compute loss between predictions and targets."""
        pass

    def configure_optimizers(self):
        """Configure optimizer for training."""
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer 