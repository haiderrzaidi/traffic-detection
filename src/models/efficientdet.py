import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

class EfficientDet(nn.Module):
    def __init__(self, num_classes, compound_coef=0):
        """
        EfficientDet model for object detection
        
        Args:
            num_classes: Number of classes (including background)
            compound_coef: Compound coefficient (0-7) for model scaling
        """
        super(EfficientDet, self).__init__()
        
        # Load EfficientNet backbone
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        
        # Remove the classifier
        self.backbone.classifier = nn.Identity()
        
        # Feature Pyramid Network (FPN)
        self.fpn_channels = 256
        self.fpn = FeaturePyramidNetwork(compound_coef)
        
        # Detection head
        self.num_classes = num_classes
        self.num_anchors = 9  # Default number of anchors
        self.detection_head = DetectionHead(
            in_channels=self.fpn_channels,
            num_classes=num_classes,
            num_anchors=self.num_anchors
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights of the model"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x, targets=None):
        """
        Forward pass of the model
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            targets: Optional target dictionary for training
            
        Returns:
            If training: classification_loss, regression_loss
            If inference: predictions
        """
        # Extract features from backbone
        features = []
        
        # First conv layer
        x = self.backbone.features[0](x)
        
        # Extract features from specific blocks
        block_indices = [2, 3, 5, 7]  # Corresponding to P3, P4, P5, P6, P7
        current_block = 0
        
        for i in range(len(self.backbone.features)):
            x = self.backbone.features[i](x)
            if i in block_indices:
                features.append(x)
                current_block += 1
        
        # Add P7 feature map
        p7 = F.max_pool2d(features[-1], kernel_size=2, stride=2)
        features.append(p7)
        
        # Apply FPN
        fpn_features = self.fpn(features)
        
        # Apply detection head
        if self.training and targets is not None:
            return self.detection_head(fpn_features, targets)
        else:
            return self.detection_head(fpn_features)
    
    def inference(self, x, score_threshold=0.05, nms_threshold=0.5):
        """
        Inference with the model
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            score_threshold: Confidence threshold for detections
            nms_threshold: NMS threshold
            
        Returns:
            List of detections for each image
        """
        self.eval()
        with torch.no_grad():
            # Forward pass
            predictions = self.forward(x)
            
            # Apply NMS
            results = []
            for pred in predictions:
                # Filter by confidence
                mask = pred[:, 4] > score_threshold
                pred = pred[mask]
                
                if len(pred) == 0:
                    results.append(torch.zeros((0, 6), device=pred.device))
                    continue
                
                # Apply NMS
                boxes = pred[:, :4]
                scores = pred[:, 4]
                labels = pred[:, 5]
                
                # Perform NMS
                keep = nms(boxes, scores, nms_threshold)
                pred = pred[keep]
                
                results.append(pred)
            
            return results

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, compound_coef):
        """
        Feature Pyramid Network for EfficientDet
        
        Args:
            compound_coef: Compound coefficient for scaling
        """
        super(FeaturePyramidNetwork, self).__init__()
        
        # Scale the number of channels based on compound coefficient
        self.channels = 256 + (compound_coef * 32)
        
        # Input channels for each level from EfficientNet-B0
        self.in_channels = [32, 48, 96, 136, 232]  # P3, P4, P5, P6, P7
        
        # Top-down pathway
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, self.channels, 1) for in_ch in self.in_channels
        ])
        
        # FPN output convolutions
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(self.channels, self.channels, 3, padding=1) for _ in range(len(self.in_channels))
        ])
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights of the FPN"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, features):
        """
        Forward pass of the FPN
        
        Args:
            features: List of feature maps from the backbone
            
        Returns:
            List of FPN feature maps
        """
        # Convert features to list if it's not already
        if not isinstance(features, list):
            features = [features]
        
        # Process each feature map with corresponding lateral convolution
        laterals = []
        for feature, lateral_conv in zip(features, self.lateral_convs):
            laterals.append(lateral_conv(feature))
        
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], 
                size=laterals[i - 1].shape[-2:],
                mode='nearest'
            )
        
        # Apply FPN convolutions
        fpn_features = []
        for lateral, fpn_conv in zip(laterals, self.fpn_convs):
            fpn_features.append(fpn_conv(lateral))
        
        return fpn_features

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors):
        """
        Detection head for EfficientDet
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of classes (including background)
            num_anchors: Number of anchors per feature map
        """
        super(DetectionHead, self).__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        )
        
        # Regression head
        self.reg_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * 4, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights of the detection head"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, features, targets=None):
        """
        Forward pass of the detection head
        
        Args:
            features: List of FPN feature maps
            targets: Optional target dictionary for training
            
        Returns:
            If training: classification_loss, regression_loss
            If inference: predictions
        """
        # Process each feature map
        cls_outputs = []
        reg_outputs = []
        
        for feature in features:
            # Classification output
            cls_output = self.cls_head(feature)
            cls_output = cls_output.permute(0, 2, 3, 1).contiguous()
            cls_output = cls_output.view(cls_output.size(0), -1, self.num_classes)
            cls_outputs.append(cls_output)
            
            # Regression output
            reg_output = self.reg_head(feature)
            reg_output = reg_output.permute(0, 2, 3, 1).contiguous()
            reg_output = reg_output.view(reg_output.size(0), -1, 4)
            reg_outputs.append(reg_output)
        
        # Concatenate outputs
        cls_outputs = torch.cat(cls_outputs, dim=1)
        reg_outputs = torch.cat(reg_outputs, dim=1)
        
        if self.training and targets is not None:
            # Calculate losses
            cls_loss = F.cross_entropy(cls_outputs.view(-1, self.num_classes), targets['labels'].view(-1))
            reg_loss = F.smooth_l1_loss(reg_outputs, targets['boxes'])
            
            return cls_loss, reg_loss
        else:
            # Return predictions
            return torch.cat([reg_outputs, cls_outputs], dim=2)

def nms(boxes, scores, iou_threshold):
    """
    Non-maximum suppression
    
    Args:
        boxes: Tensor of shape (N, 4) containing bounding boxes
        scores: Tensor of shape (N) containing scores
        iou_threshold: IoU threshold
        
    Returns:
        Indices of boxes to keep
    """
    # Sort by score
    _, order = scores.sort(0, descending=True)
    keep = []
    
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        
        i = order[0]
        keep.append(i)
        
        # Calculate IoU
        ious = box_iou(boxes[i].unsqueeze(0), boxes[order[1:]])
        
        # Filter boxes with IoU > threshold
        mask = ious.squeeze() <= iou_threshold
        order = order[1:][mask]
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

def box_iou(box1, box2):
    """
    Calculate IoU between two boxes
    
    Args:
        box1: Tensor of shape (N, 4)
        box2: Tensor of shape (M, 4)
        
    Returns:
        IoU tensor of shape (N, M)
    """
    # Calculate intersection
    lt = torch.max(box1[:, :2].unsqueeze(1), box2[:, :2].unsqueeze(0))
    rb = torch.min(box1[:, 2:].unsqueeze(1), box2[:, 2:].unsqueeze(0))
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    # Calculate union
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    union = area1.unsqueeze(1) + area2.unsqueeze(0) - inter
    
    return inter / union 