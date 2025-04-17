import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Dict, List, Tuple

from .base_model import BaseModel
from configs.model_config import FASTER_RCNN_CONFIG

class FasterRCNNModel(BaseModel):
    """Faster R-CNN model for traffic incident detection."""
    
    def __init__(
        self,
        num_classes: int = FASTER_RCNN_CONFIG['num_classes'],
        backbone: str = FASTER_RCNN_CONFIG['backbone'],
        pretrained_backbone: bool = FASTER_RCNN_CONFIG['pretrained_backbone'],
        min_size: int = FASTER_RCNN_CONFIG['min_size'],
        max_size: int = FASTER_RCNN_CONFIG['max_size'],
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        super().__init__(learning_rate=learning_rate)
        
        # Model parameters
        self.num_classes = num_classes
        
        # Load pre-trained model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True,
            min_size=min_size,
            max_size=max_size
        )
        
        # Replace the pre-trained head with a new one
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Loss function is handled internally by the model
        
        # Save hyperparameters
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_frames, channels, height, width)
        
        Returns:
            List[Dict[str, torch.Tensor]]: List of dictionaries containing detection results
        """
        batch_size, num_frames, channels, height, width = x.shape
        
        # Process each frame independently
        all_detections = []
        x = x.view(batch_size * num_frames, channels, height, width)
        
        # Set model to evaluation mode for inference
        self.model.eval()
        
        with torch.no_grad():
            # Process frames in smaller batches to avoid memory issues
            sub_batch_size = 4
            for i in range(0, len(x), sub_batch_size):
                sub_batch = x[i:i + sub_batch_size]
                detections = self.model(sub_batch)
                all_detections.extend(detections)
        
        # Reshape detections to match input batch structure
        batch_detections = []
        for i in range(batch_size):
            frame_detections = all_detections[i * num_frames:(i + 1) * num_frames]
            batch_detections.append(frame_detections)
        
        return batch_detections

    def compute_loss(self, y_hat: List[Dict[str, torch.Tensor]], y: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute Faster R-CNN loss.
        
        Note: The loss is computed internally by the model during training.
        This method is mainly for validation and testing.
        """
        # Faster R-CNN computes loss internally during training
        # For validation/testing, we compute mean of classification and regression losses
        total_loss = torch.tensor(0.0, device=self.device)
        
        for pred, target in zip(y_hat, y):
            # Classification loss
            cls_logits = pred['scores']
            cls_targets = target['labels']
            cls_loss = nn.functional.cross_entropy(cls_logits, cls_targets)
            
            # Regression loss (smooth L1 loss)
            box_regression = pred['boxes']
            box_targets = target['boxes']
            reg_loss = nn.functional.smooth_l1_loss(box_regression, box_targets)
            
            total_loss += cls_loss + reg_loss
        
        return total_loss / len(y_hat)

    def training_step(self, batch: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step."""
        images, targets = batch
        
        # Forward pass and loss computation are handled by the model
        loss_dict = self.model(images, targets)
        
        # Total loss is the sum of all losses
        total_loss = sum(loss for loss in loss_dict.values())
        
        # Log individual losses
        for name, loss in loss_dict.items():
            self.log(f'train_{name}', loss)
        
        # Log total loss
        self.log('train_loss', total_loss)
        
        return {'loss': total_loss}

    def validation_step(self, batch: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        images, targets = batch
        
        # Get model predictions
        detections = self(images)
        
        # Compute validation loss
        loss = self.compute_loss(detections, targets)
        
        # Log metrics
        self.log('val_loss', loss)
        
        return {'val_loss': loss}

    def test_step(self, batch: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        images, targets = batch
        
        # Get model predictions
        detections = self(images)
        
        # Compute test loss
        loss = self.compute_loss(detections, targets)
        
        # Log metrics
        self.log('test_loss', loss)
        
        # Calculate mean Average Precision
        pred_boxes = [d['boxes'] for d in detections]
        pred_scores = [d['scores'] for d in detections]
        pred_labels = [d['labels'] for d in detections]
        
        true_boxes = [t['boxes'] for t in targets]
        true_labels = [t['labels'] for t in targets]
        
        return {
            'test_loss': loss,
            'pred_boxes': pred_boxes,
            'pred_scores': pred_scores,
            'pred_labels': pred_labels,
            'true_boxes': true_boxes,
            'true_labels': true_labels
        }

    def configure_optimizers(self):
        """Configure optimizers for training."""
        # Use different learning rates for backbone and heads
        params = [
            {'params': self.model.backbone.parameters(), 'lr': self.learning_rate / 10},
            {'params': self.model.rpn.parameters(), 'lr': self.learning_rate},
            {'params': self.model.roi_heads.parameters(), 'lr': self.learning_rate}
        ]
        
        optimizer = torch.optim.AdamW(params, weight_decay=self.hparams.weight_decay)
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