import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List, Tuple
import torchvision.models as models

class GRUModel(pl.LightningModule):
    """GRU model for analyzing temporal patterns in accident sequences."""
    
    def __init__(
        self,
        input_size: int = 512,  # Size of feature vector from CNN backbone
        hidden_size: int = 512,
        num_layers: int = 2,
        num_classes: int = 2,
        learning_rate: float = 1e-4,
        dropout: float = 0.5,
        sequence_length: int = 16
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # CNN feature extractor
        self.cnn = models.resnet18(pretrained=True)
        # Remove the last fully connected layer
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        # Freeze CNN parameters
        for param in self.cnn.parameters():
            param.requires_grad = False
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels, height, width)
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # Reshape for CNN: (batch_size * sequence_length, channels, height, width)
        x = x.view(-1, channels, height, width)
        
        # Extract features using CNN
        features = self.cnn(x)
        features = features.view(batch_size, seq_len, -1)  # (batch_size, sequence_length, feature_size)
        
        # GRU forward pass
        gru_out, _ = self.gru(features)
        
        # Use the last output for classification
        last_output = gru_out[:, -1, :]
        
        # Pass through fully connected layers
        output = self.fc(last_output)
        return output
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        sequences, labels = batch
        outputs = self(sequences)
        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        sequences, labels = batch
        outputs = self(sequences)
        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log('val_loss', loss)
        self.log('val_acc', acc)
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        sequences, labels = batch
        outputs = self(sequences)
        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        
        return {'test_loss': loss, 'test_acc': acc}
    
    def configure_optimizers(self):
        """Configure optimizers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        } 