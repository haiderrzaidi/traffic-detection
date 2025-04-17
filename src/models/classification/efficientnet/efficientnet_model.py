import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List, Tuple
import torchvision.models as models

class EfficientNetModel(pl.LightningModule):
    """EfficientNet model for traffic incident classification."""
    
    def __init__(
        self,
        model_name: str = 'efficientnet_b0',
        pretrained: bool = True,
        num_classes: int = 2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Load pre-trained model
        self.model = getattr(models, model_name)(pretrained=pretrained)
        
        # Replace the classifier head
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len, channels, height, width = x.shape
        
        # Process each frame independently
        x = x.view(-1, channels, height, width)  # (batch_size * sequence_length, channels, height, width)
        outputs = self.model(x)
        
        # Reshape back to (batch_size, sequence_length, num_classes)
        outputs = outputs.view(batch_size, seq_len, -1)
        
        # Use the last frame's prediction
        return outputs[:, -1, :]
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        images, labels = batch
        outputs = self(images)
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
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log('val_loss', loss)
        self.log('val_acc', acc)
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        images, labels = batch
        outputs = self(images)
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
            lr=self.learning_rate,
            weight_decay=self.weight_decay
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