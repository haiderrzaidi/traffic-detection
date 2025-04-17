import os
import sys
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.configs.model_config import YOLO_CONFIG

def process_video(
    model: YOLO,
    video_path: str,
    output_dir: str,
    conf_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Process video with YOLO model and save results.
    
    Args:
        model: YOLO model
        video_path: Path to input video
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for detections
    
    Returns:
        Dictionary containing detection metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer
    output_path = os.path.join(output_dir, 'yolo_detection.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video
    frame_count = 0
    total_detections = 0
    accident_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model(frame, conf=conf_threshold)
        
        # Process detections
        for result in results:
            # Draw detections
            annotated_frame = result.plot()
            
            # Count detections
            boxes = result.boxes
            num_detections = len(boxes)
            total_detections += num_detections
            
            # Check for accidents
            if num_detections > 0:
                accident_frames += 1
            
            # Write frame
            out.write(annotated_frame)
        
        frame_count += 1
    
    # Release resources
    cap.release()
    out.release()
    
    # Calculate metrics
    metrics = {
        'total_frames': frame_count,
        'total_detections': total_detections,
        'accident_frames': accident_frames,
        'accident_ratio': accident_frames / frame_count if frame_count > 0 else 0
    }
    
    return metrics

def main():
    # Create argument parser
    import argparse
    parser = argparse.ArgumentParser(description='Run YOLO inference on video')
    parser.add_argument('--weights_path', type=str, required=True,
                      help='Path to trained model weights')
    parser.add_argument('--input_video', type=str, required=True,
                      help='Path to input video')
    parser.add_argument('--output_dir', type=str, default='output',
                      help='Directory to save results')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                      help='Confidence threshold for detections')
    
    args = parser.parse_args()
    
    # Load model
    model = YOLO(args.weights_path)
    
    # Process video
    metrics = process_video(
        model=model,
        video_path=args.input_video,
        output_dir=args.output_dir,
        conf_threshold=args.conf_threshold
    )
    
    # Print metrics
    print("\nDetection Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

if __name__ == '__main__':
    main() 