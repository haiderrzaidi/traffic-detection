import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List, Tuple
import torchvision.transforms as T
from effdet import create_model, create_model_from_config
from effdet.config.model_config import efficientdet_model_param_dict

class EfficientDetModel(pl.LightningModule):
    """EfficientDet model wrapper for optimized accident detection."""
    
    def __init__(
        self,
        model_name: str = "tf_efficientdet_d0",
        num_classes: int = 2,  # Binary classification: accident or no accident
        pretrained: bool = True,
        learning_rate: float = 1e-4,
        confidence_threshold: float = 0.5
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Load EfficientDet model
        self.model = create_model(
            model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            bench_task='train'
        )
        self.confidence_threshold = confidence_threshold
        self.learning_rate = learning_rate
        
        # Define transforms
        self.transform = T.Compose([
            T.Resize((512, 512)),  # EfficientDet-D0 default size
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of the model."""
        return self.model(x)
    
    def training_step(self, batch: Tuple[torch.Tensor, Dict], batch_idx: int) -> torch.Tensor:
        """Training step."""
        images, targets = batch
        outputs = self.model(images, targets)
        loss = outputs['loss']
        
        # Log individual losses
        for k, v in outputs.items():
            if 'loss' in k:
                self.log(f"train_{k}", v)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, Dict], batch_idx: int) -> None:
        """Validation step."""
        images, targets = batch
        outputs = self.model(images, targets)
        
        # Log validation metrics
        for k, v in outputs.items():
            if 'loss' in k:
                self.log(f"val_{k}", v)
    
    def configure_optimizers(self):
        """Configure optimizers."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
    
    def detect_accidents(self, frame: torch.Tensor) -> List[Dict]:
        """Detect accidents in a single frame."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(frame)
            
        detections = []
        scores = outputs[0]['scores']
        boxes = outputs[0]['boxes']
        labels = outputs[0]['labels']
        
        # Filter detections by confidence threshold
        mask = scores > self.confidence_threshold
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        filtered_labels = labels[mask]
        
        for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
            detection = {
                "bbox": box.tolist(),
                "confidence": score.item(),
                "class": label.item(),
                "class_name": "accident" if label.item() == 1 else "normal"
            }
            detections.append(detection)
        
        return detections 