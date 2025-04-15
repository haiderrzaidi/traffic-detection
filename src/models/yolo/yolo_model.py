import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List, Tuple
import torchvision.transforms as T
from ultralytics import YOLO

class YOLOModel(pl.LightningModule):
    """YOLOv8 model wrapper for accident detection."""
    
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        pretrained: bool = True,
        learning_rate: float = 1e-4,
        confidence_threshold: float = 0.5
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Load YOLOv8 model
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.learning_rate = learning_rate
        
        # Define transforms
        self.transform = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of the model."""
        return self.model(x)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        images, targets = batch
        loss_dict = self.model.train(images, targets)
        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()})
        return loss_dict["loss"]
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Validation step."""
        images, targets = batch
        loss_dict = self.model.val(images, targets)
        self.log_dict({f"val_{k}": v for k, v in loss_dict.items()})
    
    def configure_optimizers(self):
        """Configure optimizers."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
    
    def detect_accidents(self, frame: torch.Tensor) -> List[Dict]:
        """Detect accidents in a single frame."""
        results = self.model(frame, conf=self.confidence_threshold)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    "bbox": box.xyxy[0].tolist(),  # Convert to list for JSON serialization
                    "confidence": box.conf.item(),
                    "class": box.cls.item(),
                    "class_name": result.names[int(box.cls.item())]
                }
                detections.append(detection)
        
        return detections 