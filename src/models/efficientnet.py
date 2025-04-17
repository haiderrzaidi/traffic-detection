import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Tuple

from src.models.base_model import BaseModel

class EfficientNetModel(BaseModel):
    """EfficientNet model for traffic incident classification."""
    
    def __init__(
        self,
        model_name: str = 'efficientnet_b0',
        pretrained: bool = True,
        num_classes: int = 2,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        super().__init__(learning_rate=learning_rate)
        
        # Load pretrained EfficientNet
        self.backbone = getattr(models, model_name)(pretrained=pretrained)
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add temporal pooling and classification head
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),  # EfficientNet B0 features are 1280-dimensional
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_frames, channels, height, width)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        batch_size, num_frames, channels, height, width = x.shape
        
        # Reshape input to process all frames
        x = x.view(batch_size * num_frames, channels, height, width)
        
        # Extract features for each frame
        features = self.features(x)  # Shape: (batch_size * num_frames, 1280, 1, 1)
        
        # Reshape features to (batch_size, num_frames, 1280)
        features = features.view(batch_size, num_frames, 1280)
        
        # Add spatial dimensions for 3D pooling
        features = features.unsqueeze(3).unsqueeze(4)  # Shape: (batch_size, num_frames, 1280, 1, 1)
        
        # Permute to (batch_size, 1280, num_frames, 1, 1) for temporal pooling
        features = features.permute(0, 2, 1, 3, 4)
        
        # Apply temporal pooling
        pooled = self.temporal_pool(features)  # Shape: (batch_size, 1280, 1, 1, 1)
        
        # Flatten and classify
        pooled = pooled.view(batch_size, 1280)
        output = self.classifier(pooled)
        
        return output

    def compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the loss for a batch."""
        return self.criterion(y_hat, y)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step."""
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        
        # Compute accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return {'loss': loss, 'acc': acc}

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        
        # Compute accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        
        # Compute accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        
        return {'test_loss': loss, 'test_acc': acc} 