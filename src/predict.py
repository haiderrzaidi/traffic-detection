import os
import cv2
import torch
import numpy as np
from typing import List, Tuple, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.configs.model_config import DATASET_CONFIG
from src.models.yolo.yolo_model import YOLOModel
from src.models.resnet.resnet_model import ResNetModel
from src.models.lstm.lstm_model import LSTMModel
from src.models.ensemble.ensemble_model import EnsembleModel

class TrafficIncidentPredictor:
    def __init__(self, model_paths: Dict[str, str], device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the predictor with trained models.
        
        Args:
            model_paths: Dictionary containing paths to trained model checkpoints
            device: Device to run inference on
        """
        self.device = device
        self.transform = A.Compose([
            A.Resize(height=DATASET_CONFIG['image_size'][0], width=DATASET_CONFIG['image_size'][1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        # Load models
        self.models = {}
        for model_name, path in model_paths.items():
            if model_name == 'yolo':
                model = YOLOModel.load_from_checkpoint(path)
            elif model_name == 'resnet':
                model = ResNetModel.load_from_checkpoint(path)
            elif model_name == 'lstm':
                model = LSTMModel.load_from_checkpoint(path)
            else:
                raise ValueError(f"Unknown model type: {model_name}")
            
            model.to(device)
            model.eval()
            self.models[model_name] = model
        
        # Initialize ensemble model
        self.ensemble = EnsembleModel(self.models)
        self.ensemble.to(device)
        self.ensemble.eval()

    def preprocess_video(self, video_path: str) -> torch.Tensor:
        """
        Preprocess a video file into a tensor of frames.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tensor of shape (T, C, H, W)
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to sample
        frame_indices = np.linspace(0, total_frames-1, DATASET_CONFIG['sequence_length'], dtype=int)
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in frame_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                transformed = self.transform(image=frame)['image']
                frames.append(transformed)
        
        cap.release()
        
        if len(frames) < DATASET_CONFIG['sequence_length']:
            # Pad with last frame if not enough frames
            last_frame = frames[-1]
            while len(frames) < DATASET_CONFIG['sequence_length']:
                frames.append(last_frame)
        
        return torch.stack(frames)

    def predict(self, video_path: str) -> Tuple[float, Dict[str, float]]:
        """
        Make predictions on a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (ensemble_score, individual_scores)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Preprocess video
        frames = self.preprocess_video(video_path)
        frames = frames.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Get predictions from individual models
        individual_scores = {}
        with torch.no_grad():
            for model_name, model in self.models.items():
                if model_name == 'yolo':
                    # YOLO model expects different input format
                    score = model(frames[0])  # Remove batch dimension for YOLO
                else:
                    score = model(frames)
                individual_scores[model_name] = score.item()
            
            # Get ensemble prediction
            ensemble_score = self.ensemble(frames).item()
        
        return ensemble_score, individual_scores

    def predict_batch(self, video_paths: List[str]) -> List[Tuple[float, Dict[str, float]]]:
        """
        Make predictions on multiple video files.
        
        Args:
            video_paths: List of paths to video files
            
        Returns:
            List of (ensemble_score, individual_scores) tuples
        """
        return [self.predict(path) for path in video_paths]

def main():
    # Example usage
    model_paths = {
        'yolo': 'checkpoints/yolo.ckpt',
        'resnet': 'checkpoints/resnet.ckpt',
        'lstm': 'checkpoints/lstm.ckpt'
    }
    
    predictor = TrafficIncidentPredictor(model_paths)
    
    # Example video path
    video_path = "path/to/video.mp4"
    
    try:
        ensemble_score, individual_scores = predictor.predict(video_path)
        print(f"Ensemble Score: {ensemble_score:.4f}")
        print("Individual Model Scores:")
        for model_name, score in individual_scores.items():
            print(f"{model_name}: {score:.4f}")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main() 