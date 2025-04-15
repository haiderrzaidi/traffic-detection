import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List, Tuple
import torchvision.transforms as T

class LSTMModel(pl.LightningModule):
    """LSTM model for analyzing temporal patterns in accident sequences."""
    
    def __init__(
        self,
        input_size: int = 2048,  # Size of feature vector from backbone
        hidden_size: int = 512,
        num_layers: int = 2,
        num_classes: int = 2,
        learning_rate: float = 1e-4,
        dropout: float = 0.5,
        sequence_length: int = 16
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # LSTM layers
        self.lstm = nn.LSTM(
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
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Use the last output for classification
        last_output = lstm_out[:, -1, :]
        
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
    
    def configure_optimizers(self):
        """Configure optimizers."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
    
    def predict_sequence(self, sequence: torch.Tensor) -> Dict:
        """Predict accident probability for a sequence of frames.
        
        Args:
            sequence: Tensor of shape (1, sequence_length, input_size)
        Returns:
            Dictionary containing prediction probabilities
        """
        self.eval()
        with torch.no_grad():
            output = self(sequence)
            probabilities = torch.softmax(output, dim=1)
            
        return {
            "accident_prob": probabilities[0, 1].item(),
            "normal_prob": probabilities[0, 0].item(),
            "prediction": "accident" if probabilities[0, 1] > 0.5 else "normal"
        } 