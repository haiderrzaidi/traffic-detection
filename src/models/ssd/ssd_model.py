import torch
import torch.nn as nn
import torchvision.models.detection as detection
from typing import Dict, List, Tuple
from torchvision.models.detection.ssd import SSD300_VGG16_Weights

from src.models.base_model import BaseModel

class SSDModel(BaseModel):
    """SSD (Single Shot MultiBox Detector) model for car collision detection."""
    
    def __init__(
        self,
        num_classes: int = 2,  # Background + Car
        learning_rate: float = 1e-4,
        pretrained: bool = True,
        pretrained_backbone: bool = True
    ):
        super().__init__(learning_rate=learning_rate)
        self.save_hyperparameters()
        
        # Create SSD model with COCO weights first
        if pretrained:
            weights = SSD300_VGG16_Weights.COCO_V1
        else:
            weights = None
            
        self.model = detection.ssd300_vgg16(weights=weights)
        
        # Replace the classifier head with a new one for our number of classes
        if num_classes != 91:  # 91 is the number of COCO classes
            # Get the number of anchors from the anchor generator
            num_anchors = self.model.anchor_generator.num_anchors_per_location()
            
            # Get the number of input channels from the backbone
            # For VGG16 backbone, the output channels are [512, 1024, 512, 256, 256, 256]
            in_channels = [512, 1024, 512, 256, 256, 256]
            
            # Create new classification head
            self.model.head.classification_head = detection.ssd.SSDClassificationHead(
                in_channels=in_channels,
                num_anchors=num_anchors,
                num_classes=num_classes
            )
        
        # Freeze backbone if using pretrained
        if pretrained_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        
    def forward(self, images: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.
        
        Args:
            images: List of input images
            targets: Optional list of target dictionaries with 'boxes' and 'labels'
        Returns:
            Dict of losses during training, or detections during inference
        """
        self.model.train(self.training)
        
        if self.training and targets is not None:
            outputs = self.model(images, targets)
            # Ensure we always return a dictionary of losses
            if not isinstance(outputs, dict):
                # If outputs is not a dictionary, wrap it in a dictionary
                return {'loss': outputs}
            return outputs
        else:
            detections = self.model(images)
            # During inference, return a list of dictionaries
            if isinstance(detections, list):
                return detections
            # If it's not a list, wrap it in a list
            return [detections]
    
    def training_step(self, batch: Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]], batch_idx: int) -> torch.Tensor:
        """Training step."""
        images, targets = batch
        loss_dict = self(images, targets)
        
        # Handle different loss dictionary formats
        if isinstance(loss_dict, dict):
            # Calculate total loss from dictionary
            losses = sum(loss for loss in loss_dict.values())
            
            # Log individual losses
            for loss_name, loss_value in loss_dict.items():
                # Convert tensor to scalar for logging
                if isinstance(loss_value, torch.Tensor):
                    loss_value = loss_value.item()
                self.log(f'train_{loss_name}', loss_value)
        else:
            # If it's not a dictionary, assume it's a tensor
            losses = loss_dict
            if isinstance(losses, torch.Tensor):
                losses = losses.item()
        
        self.log('train_loss', losses)
        return losses
    
    def validation_step(self, batch: Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]], batch_idx: int) -> None:
        """Validation step."""
        images, targets = batch
        loss_dict = self(images, targets)
        
        # Handle different loss dictionary formats
        if isinstance(loss_dict, dict):
            # Calculate total loss from dictionary
            losses = sum(loss for loss in loss_dict.values())
            
            # Log individual losses
            for loss_name, loss_value in loss_dict.items():
                # Convert tensor to scalar for logging
                if isinstance(loss_value, torch.Tensor):
                    loss_value = loss_value.item()
                self.log(f'val_{loss_name}', loss_value)
        else:
            # If it's not a dictionary, assume it's a tensor
            losses = loss_dict
            if isinstance(losses, torch.Tensor):
                losses = losses.item()
        
        self.log('val_loss', losses)
    
    def test_step(self, batch: Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        images, targets = batch
        loss_dict = self(images, targets)
        
        # Handle different loss dictionary formats
        if isinstance(loss_dict, dict):
            # Calculate total loss from dictionary
            losses = sum(loss for loss in loss_dict.values())
            
            # Log individual losses
            for loss_name, loss_value in loss_dict.items():
                # Convert tensor to scalar for logging
                if isinstance(loss_value, torch.Tensor):
                    loss_value = loss_value.item()
                self.log(f'test_{loss_name}', loss_value)
        else:
            # If it's not a dictionary, assume it's a tensor
            losses = loss_dict
            if isinstance(losses, torch.Tensor):
                losses = losses.item()
        
        self.log('test_loss', losses)
        return {'test_loss': losses}
    
    def configure_optimizers(self):
        """Configure optimizers."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
    
    def predict_image(self, image: torch.Tensor, confidence_threshold: float = 0.5) -> Dict:
        """Predict objects in a single image.
        
        Args:
            image: Input image tensor of shape (C, H, W)
            confidence_threshold: Minimum confidence score for detections
        Returns:
            Dictionary containing detections
        """
        self.eval()
        with torch.no_grad():
            predictions = self([image])[0]
            
            # Filter predictions by confidence
            keep = predictions['scores'] > confidence_threshold
            filtered_predictions = {
                'boxes': predictions['boxes'][keep],
                'labels': predictions['labels'][keep],
                'scores': predictions['scores'][keep]
            }
            
        return filtered_predictions 