import cv2
import torch
from ultralytics import YOLO
import argparse
from pathlib import Path
import time
import numpy as np
from typing import List, Tuple, Dict
import json

def process_image(
    model: YOLO,
    image_path: str,
    conf_threshold: float = 0.25,
    save_output: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Process a single image and return the annotated image and detection metrics.
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Run inference
    start_time = time.time()
    results = model(image, conf=conf_threshold)[0]
    inference_time = time.time() - start_time
    
    # Get detections
    detections = results.boxes.data.cpu().numpy()
    
    # Calculate metrics
    metrics = {
        'inference_time': inference_time,
        'num_detections': len(detections),
        'classes_detected': list(set(detections[:, -1].astype(int))) if len(detections) > 0 else [],
        'confidence_scores': detections[:, 4].tolist() if len(detections) > 0 else []
    }
    
    # Draw detections
    annotated_image = results.plot()
    
    # Save output if requested
    if save_output:
        output_path = str(Path(image_path).with_name(Path(image_path).stem + '_detected.jpg'))
        cv2.imwrite(output_path, annotated_image)
        metrics['output_path'] = output_path
    
    return annotated_image, metrics

def process_video(
    model: YOLO,
    video_path: str,
    output_path: str = None,
    conf_threshold: float = 0.25,
    save_video: bool = True
) -> Dict:
    """
    Process a video file and return performance metrics.
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output video writer if needed
    if save_video:
        if output_path is None:
            output_path = str(Path(video_path).with_name(Path(video_path).stem + '_detected.mp4'))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize metrics
    metrics = {
        'total_frames': total_frames,
        'processed_frames': 0,
        'total_inference_time': 0,
        'fps': [],
        'detections_per_frame': [],
        'classes_detected': set()
    }
    
    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        start_time = time.time()
        results = model(frame, conf=conf_threshold)[0]
        inference_time = time.time() - start_time
        
        # Update metrics
        detections = results.boxes.data.cpu().numpy()
        metrics['total_inference_time'] += inference_time
        metrics['fps'].append(1.0 / inference_time if inference_time > 0 else 0)
        metrics['detections_per_frame'].append(len(detections))
        metrics['classes_detected'].update(set(detections[:, -1].astype(int)) if len(detections) > 0 else [])
        
        # Draw detections
        annotated_frame = results.plot()
        
        # Save frame if needed
        if save_video:
            out.write(annotated_frame)
        
        # Display frame
        cv2.imshow('YOLOv8 Inference', annotated_frame)
        
        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        metrics['processed_frames'] += 1
    
    # Cleanup
    cap.release()
    if save_video:
        out.release()
    cv2.destroyAllWindows()
    
    # Calculate final metrics
    metrics['average_fps'] = np.mean(metrics['fps'])
    metrics['average_detections'] = np.mean(metrics['detections_per_frame'])
    metrics['classes_detected'] = list(metrics['classes_detected'])
    if save_video:
        metrics['output_path'] = output_path
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Test YOLOv8 inference on images or videos')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained YOLOv8 model')
    parser.add_argument('--input_path', type=str, required=True,
                      help='Path to input image or video file')
    parser.add_argument('--output_path', type=str, default=None,
                      help='Path to save output (optional)')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                      help='Confidence threshold for detections')
    parser.add_argument('--no_save', action='store_true',
                      help='Do not save output')
    
    args = parser.parse_args()
    
    # Initialize model
    model = YOLO(args.model_path)
    
    # Determine if input is image or video
    input_path = Path(args.input_path)
    if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        # Process image
        _, metrics = process_image(
            model=model,
            image_path=str(input_path),
            conf_threshold=args.conf_threshold,
            save_output=not args.no_save
        )
        print("\nImage Processing Metrics:")
    else:
        # Process video
        metrics = process_video(
            model=model,
            video_path=str(input_path),
            output_path=args.output_path,
            conf_threshold=args.conf_threshold,
            save_video=not args.no_save
        )
        print("\nVideo Processing Metrics:")
    
    # Print metrics
    print(json.dumps(metrics, indent=2))
    
    # Save metrics to file
    metrics_path = str(Path(args.input_path).with_name(Path(args.input_path).stem + '_metrics.json'))
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

if __name__ == '__main__':
    main() 