import torch
import torch.nn as nn
from typing import Dict, Tuple
import torchvision.ops as ops

from .base_model import BaseModel
from ..configs.model_config import YOLO_CONFIG

class ConvBlock(nn.Module):
    """Basic convolution block used in YOLOv7."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class CSPBlock(nn.Module):
    """Cross Stage Partial Network block used in YOLOv7."""
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 1):
        super().__init__()
        mid_channels = out_channels // 2
        
        self.conv1 = ConvBlock(in_channels, mid_channels, 1)
        self.conv2 = ConvBlock(in_channels, mid_channels, 1)
        self.conv3 = ConvBlock(2 * mid_channels, out_channels, 1)
        
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                ConvBlock(mid_channels, mid_channels, 1),
                ConvBlock(mid_channels, mid_channels, 3)
            ) for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.blocks(self.conv2(x))
        return self.conv3(torch.cat([x1, x2], dim=1))

class YOLOModel(BaseModel):
    """YOLOv7 model for traffic incident detection."""
    
    def __init__(
        self,
        num_classes: int = 2,  # background + accident
        input_size: Tuple[int, int] = YOLO_CONFIG['input_size'],
        confidence_threshold: float = YOLO_CONFIG['confidence_threshold'],
        nms_threshold: float = YOLO_CONFIG['nms_threshold'],
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        super().__init__(learning_rate=learning_rate)
        
        # Model parameters
        self.num_classes = num_classes
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Backbone
        self.backbone = nn.ModuleList([
            # Initial conv
            ConvBlock(3, 32, 3),
            
            # Downsample 1: 32 -> 64
            ConvBlock(32, 64, 3, stride=2),
            CSPBlock(64, 64, num_blocks=1),
            
            # Downsample 2: 64 -> 128
            ConvBlock(64, 128, 3, stride=2),
            CSPBlock(128, 128, num_blocks=2),
            
            # Downsample 3: 128 -> 256
            ConvBlock(128, 256, 3, stride=2),
            CSPBlock(256, 256, num_blocks=3),
            
            # Downsample 4: 256 -> 512
            ConvBlock(256, 512, 3, stride=2),
            CSPBlock(512, 512, num_blocks=3),
            
            # Downsample 5: 512 -> 1024
            ConvBlock(512, 1024, 3, stride=2),
            CSPBlock(1024, 1024, num_blocks=3),
        ])
        
        # Detection head
        self.head = nn.Sequential(
            ConvBlock(1024, 512, 1),
            ConvBlock(512, 1024, 3),
            ConvBlock(1024, 512, 1),
            nn.Conv2d(512, (num_classes + 5) * 3, 1)  # 3 anchors per cell
        )
        
        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        
        # Initialize weights
        self._init_weights()
        
        # Save hyperparameters
        self.save_hyperparameters()

    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_frames, channels, height, width)
        
        Returns:
            torch.Tensor: Output tensor with detection predictions
        """
        batch_size, num_frames, channels, height, width = x.shape
        
        # Process each frame independently
        outputs = []
        x = x.view(batch_size * num_frames, channels, height, width)
        
        # Backbone features
        features = x
        for layer in self.backbone:
            features = layer(features)
        
        # Detection head
        detections = self.head(features)
        
        # Reshape output
        detections = detections.view(batch_size, num_frames, -1, height//32, width//32)
        
        return detections

    def compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute YOLO loss."""
        # Unpack predictions
        pred_obj = y_hat[..., 4]
        pred_cls = y_hat[..., 5:]
        pred_box = y_hat[..., :4]
        
        # Unpack targets
        true_obj = y[..., 4]
        true_cls = y[..., 5:]
        true_box = y[..., :4]
        
        # Object loss
        obj_loss = self.bce_loss(pred_obj, true_obj)
        
        # Class loss (only for positive samples)
        cls_loss = self.bce_loss(pred_cls[true_obj > 0], true_cls[true_obj > 0])
        
        # Box loss (only for positive samples)
        box_loss = self.mse_loss(pred_box[true_obj > 0], true_box[true_obj > 0])
        
        # Total loss
        total_loss = obj_loss + cls_loss + box_loss
        
        return total_loss

    def _post_process_detections(
        self,
        detections: torch.Tensor,
        conf_threshold: float = None,
        nms_threshold: float = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Post-process YOLO detections with NMS."""
        if conf_threshold is None:
            conf_threshold = self.confidence_threshold
        if nms_threshold is None:
            nms_threshold = self.nms_threshold
            
        batch_size = detections.shape[0]
        
        # Process each image in batch
        output_boxes = []
        output_scores = []
        output_labels = []
        
        for i in range(batch_size):
            # Get detections for this image
            img_dets = detections[i]
            
            # Get confidence scores
            scores = img_dets[..., 4] * img_dets[..., 5:].max(dim=-1)[0]
            
            # Filter by confidence
            mask = scores > conf_threshold
            boxes = img_dets[mask, :4]
            scores = scores[mask]
            labels = img_dets[mask, 5:].max(dim=-1)[1]
            
            # Apply NMS
            keep = ops.nms(boxes, scores, nms_threshold)
            
            output_boxes.append(boxes[keep])
            output_scores.append(scores[keep])
            output_labels.append(labels[keep])
        
        return output_boxes, output_scores, output_labels

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step."""
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        
        # Log metrics
        self.log('train_loss', loss)
        
        return {'loss': loss}

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        
        # Log metrics
        self.log('val_loss', loss)
        
        return {'val_loss': loss}

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        
        # Post-process detections
        boxes, scores, labels = self._post_process_detections(y_hat)
        
        # Log metrics
        self.log('test_loss', loss)
        
        return {
            'test_loss': loss,
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        } 