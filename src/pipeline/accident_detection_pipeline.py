import torch
import cv2
import numpy as np
from typing import Dict, List, Optional
from collections import deque
import threading
import queue
import time

from src.models.yolo.yolo_model import YOLOModel
from src.models.efficientdet.efficientdet_model import EfficientDetModel
from src.models.lstm.lstm_model import LSTMModel

class AccidentDetectionPipeline:
    """Pipeline combining YOLOv7, EfficientDet, and LSTM for real-time accident detection."""
    
    def __init__(
        self,
        yolo_checkpoint: Optional[str] = None,
        efficientdet_checkpoint: Optional[str] = None,
        lstm_checkpoint: Optional[str] = None,
        sequence_length: int = 16,
        confidence_threshold: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        
        # Initialize models
        self.yolo_model = YOLOModel().to(device)
        self.efficientdet_model = EfficientDetModel().to(device)
        self.lstm_model = LSTMModel().to(device)
        
        # Load checkpoints if provided
        if yolo_checkpoint:
            self.yolo_model.load_state_dict(torch.load(yolo_checkpoint))
        if efficientdet_checkpoint:
            self.efficientdet_model.load_state_dict(torch.load(efficientdet_checkpoint))
        if lstm_checkpoint:
            self.lstm_model.load_state_dict(torch.load(lstm_checkpoint))
        
        # Set models to evaluation mode
        self.yolo_model.eval()
        self.efficientdet_model.eval()
        self.lstm_model.eval()
        
        # Initialize frame buffer for sequence analysis
        self.frame_buffer = deque(maxlen=sequence_length)
        
        # Initialize queues for parallel processing
        self.frame_queue = queue.Queue(maxsize=32)
        self.result_queue = queue.Queue()
        
        # Start processing thread
        self.running = True
        self.process_thread = threading.Thread(target=self._process_frames)
        self.process_thread.start()
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for model input."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        frame_tensor = self.yolo_model.transform(image=frame_rgb)['image']
        frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
        
        return frame_tensor
    
    def _process_frames(self):
        """Process frames in a separate thread."""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                result = self.process_single_frame(frame)
                self.result_queue.put(result)
                self.frame_queue.task_done()
            except queue.Empty:
                continue
    
    def process_single_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame through all models."""
        frame_tensor = self.preprocess_frame(frame)
        
        # Get detections from both object detection models
        yolo_detections = self.yolo_model.detect_accidents(frame_tensor)
        efficientdet_detections = self.efficientdet_model.detect_accidents(frame_tensor)
        
        # Combine detections using non-maximum suppression
        combined_detections = self._combine_detections(yolo_detections, efficientdet_detections)
        
        # Add frame features to buffer for sequence analysis
        with torch.no_grad():
            frame_features = self.extract_features(frame_tensor)
            self.frame_buffer.append(frame_features)
        
        # If we have enough frames, perform sequence analysis
        sequence_prediction = None
        if len(self.frame_buffer) == self.sequence_length:
            sequence = torch.stack(list(self.frame_buffer)).unsqueeze(0)
            sequence_prediction = self.lstm_model.predict_sequence(sequence)
        
        return {
            "detections": combined_detections,
            "sequence_prediction": sequence_prediction,
            "timestamp": time.time()
        }
    
    def _combine_detections(self, yolo_dets: List[Dict], effdet_dets: List[Dict]) -> List[Dict]:
        """Combine detections from both models using non-maximum suppression."""
        all_boxes = []
        all_scores = []
        all_labels = []
        
        # Combine detections from both models
        for det in yolo_dets + effdet_dets:
            all_boxes.append(torch.tensor(det["bbox"]))
            all_scores.append(det["confidence"])
            all_labels.append(det["class"])
        
        if not all_boxes:
            return []
        
        # Convert to tensors
        boxes = torch.stack(all_boxes)
        scores = torch.tensor(all_scores)
        labels = torch.tensor(all_labels)
        
        # Apply NMS
        keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)
        
        # Create final detection list
        final_detections = []
        for idx in keep:
            final_detections.append({
                "bbox": boxes[idx].tolist(),
                "confidence": scores[idx].item(),
                "class": labels[idx].item(),
                "class_name": "accident" if labels[idx].item() == 1 else "normal"
            })
        
        return final_detections
    
    def extract_features(self, frame_tensor: torch.Tensor) -> torch.Tensor:
        """Extract features from frame for sequence analysis."""
        # Use EfficientDet's backbone as feature extractor
        features = self.efficientdet_model.model.backbone(frame_tensor)
        # Global average pooling to get fixed-size feature vector
        features = torch.nn.functional.adaptive_avg_pool2d(features[-1], (1, 1))
        return features.squeeze()
    
    def process_video_stream(self, video_source: int = 0):
        """Process video stream in real-time."""
        cap = cv2.VideoCapture(video_source)
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add frame to processing queue
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                
                # Get and display results
                try:
                    result = self.result_queue.get_nowait()
                    self._display_results(frame, result)
                    cv2.imshow('Accident Detection', frame)
                except queue.Empty:
                    cv2.imshow('Accident Detection', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            self.running = False
            self.process_thread.join()
            cap.release()
            cv2.destroyAllWindows()
    
    def _display_results(self, frame: np.ndarray, result: Dict):
        """Display detection results on frame."""
        # Draw bounding boxes
        for det in result["detections"]:
            bbox = det["bbox"]
            conf = det["confidence"]
            label = det["class_name"]
            
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 0, 255) if label == "accident" else (0, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display sequence prediction
        if result["sequence_prediction"]:
            pred = result["sequence_prediction"]
            text = f"Accident Probability: {pred['accident_prob']:.2f}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2) 