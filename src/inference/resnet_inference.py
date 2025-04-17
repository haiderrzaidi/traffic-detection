import os
import sys
import argparse
import torch
import cv2
import numpy as np
from typing import List, Tuple
from PIL import Image

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.resnet import ResNetModel
from src.configs.model_config import RESNET_CONFIG, DATASET_CONFIG

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with trained ResNet model')
    parser.add_argument('--weights_path', type=str, required=True,
                      help='Path to trained model weights')
    parser.add_argument('--input_video', type=str, required=True,
                      help='Path to input video file')
    parser.add_argument('--output_dir', type=str, default='outputs',
                      help='Directory to save outputs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to run inference on')
    return parser.parse_args()

def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """Preprocess video frame for model input."""
    # Resize frame
    frame = cv2.resize(frame, DATASET_CONFIG['image_size'])
    
    # Convert to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Normalize
    frame = frame.astype(np.float32) / 255.0
    frame = (frame - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # Convert to tensor and ensure float32 type
    frame = torch.from_numpy(frame).permute(2, 0, 1).float()
    
    return frame

def get_frames(video_path: str, sequence_length: int) -> List[np.ndarray]:
    """Extract frames from video file."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    while len(frames) < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    # If we don't have enough frames, duplicate the last frame
    while len(frames) < sequence_length:
        frames.append(frames[-1])
    
    return frames

def run_inference(
    model: ResNetModel,
    frames: List[np.ndarray],
    device: str
) -> Tuple[int, float]:
    """Run inference on a sequence of frames."""
    # Preprocess frames
    processed_frames = torch.stack([preprocess_frame(frame) for frame in frames])
    processed_frames = processed_frames.unsqueeze(0)  # Add batch dimension
    
    # Move to device and ensure float32 type
    processed_frames = processed_frames.to(device).float()
    
    # Run inference
    with torch.no_grad():
        outputs = model(processed_frames)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0, prediction].item()
    
    return prediction, confidence

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    model = ResNetModel(
        num_classes=RESNET_CONFIG['num_classes'],
        pretrained=False,
        learning_rate=1e-4,
        weight_decay=1e-5
    )
    
    # Load weights
    model.load_state_dict(torch.load(args.weights_path))
    model = model.to(args.device)
    model.eval()
    
    # Get video frames
    frames = get_frames(args.input_video, DATASET_CONFIG['sequence_length'])
    
    # Run inference
    prediction, confidence = run_inference(model, frames, args.device)
    
    # Print results
    class_names = ['No Accident', 'Minor Accident', 'Major Accident']
    print(f"\nPrediction: {class_names[prediction]}")
    print(f"Confidence: {confidence:.2%}")
    
    # Save visualization
    output_path = os.path.join(args.output_dir, 'prediction.jpg')
    frame = frames[len(frames)//2]  # Use middle frame for visualization
    
    # Add prediction text to frame
    text = f"{class_names[prediction]} ({confidence:.2%})"
    cv2.putText(
        frame, text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
    
    # Save frame
    cv2.imwrite(output_path, frame)
    print(f"\nVisualization saved to: {output_path}")

if __name__ == '__main__':
    main() 