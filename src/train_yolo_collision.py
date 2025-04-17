import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch
from typing import Dict, Any
import random

def train_yolo_collision(
    data_yaml: str,
    weights_path: str = "yolov8n.pt",
    output_dir: str = "runs/train",
    epochs: int = 1,
    batch_size: int = 16,
    img_size: int = 640,
    data_fraction: float = 0.3
) -> Dict[str, Any]:
    """
    Train YOLO model on car collision dataset.
    
    Args:
        data_yaml: Path to dataset YAML file
        weights_path: Path to pre-trained weights
        output_dir: Directory to save training outputs
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        data_fraction: Fraction of dataset to use (0.0 to 1.0)
    
    Returns:
        Dictionary containing training metrics
    """
    # Initialize model
    model = YOLO(weights_path)
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=output_dir,
        name='yolo_collision_training',
        exist_ok=True,
        fraction=data_fraction  # Use only specified fraction of the dataset
    )
    
    # Get metrics - using the correct path format for YOLOv8
    metrics = {
        'box_loss': results.results_dict.get('train/box_loss', 0.0),
        'cls_loss': results.results_dict.get('train/cls_loss', 0.0),
        'dfl_loss': results.results_dict.get('train/dfl_loss', 0.0),
        'mAP50': results.results_dict.get('metrics/mAP50(B)', 0.0),
        'mAP50-95': results.results_dict.get('metrics/mAP50-95(B)', 0.0)
    }
    
    return metrics

def main():
    # Create argument parser
    import argparse
    parser = argparse.ArgumentParser(description='Train YOLO model on car collision dataset')
    parser.add_argument('--data_yaml', type=str, required=True,
                      help='Path to dataset YAML file')
    parser.add_argument('--weights_path', type=str, default='yolov8n.pt',
                      help='Path to pre-trained weights')
    parser.add_argument('--output_dir', type=str, default='runs/train',
                      help='Directory to save training outputs')
    parser.add_argument('--epochs', type=int, default=1,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=640,
                      help='Input image size')
    parser.add_argument('--data_fraction', type=float, default=0.3,
                      help='Fraction of dataset to use (0.0 to 1.0)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train model
    metrics = train_yolo_collision(
        data_yaml=args.data_yaml,
        weights_path=args.weights_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        data_fraction=args.data_fraction
    )
    
    # Print metrics
    print("\nTraining Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    main() 