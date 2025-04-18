import cv2
import torch
from ultralytics import YOLO
import argparse
from pathlib import Path
import time
import numpy as np
import json
import os

def process_image(
    model: YOLO,
    image_path: str,
    output_dir: str = "results",
    conf_threshold: float = 0.05,
    save_output: bool = True,
    display: bool = False
):
    """
    Process a single image for collision detection.
    
    Args:
        model: YOLOv8 model
        image_path: Path to input image
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for detections
        save_output: Whether to save the output image
        display: Whether to display the image (requires GUI support)
    
    Returns:
        Dictionary with detection results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Draw detections
    annotated_image = results.plot()
    
    # Display image if requested and GUI is available
    if display:
        try:
            cv2.imshow('YOLOv8 Collision Detection', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as e:
            print(f"Warning: Could not display image. GUI may not be available: {e}")
    
    # Save output if requested
    output_path = None
    if save_output:
        output_filename = f"{Path(image_path).stem}_detected{Path(image_path).suffix}"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, annotated_image)
    
    # Prepare results
    result = {
        'input_path': image_path,
        'output_path': output_path,
        'inference_time': inference_time,
        'num_detections': len(detections),
        'classes_detected': list(set(detections[:, -1].astype(int))) if len(detections) > 0 else [],
        'confidence_scores': detections[:, 4].tolist() if len(detections) > 0 else []
    }
    
    # Save results to JSON
    result_filename = f"{Path(image_path).stem}_results.json"
    result_path = os.path.join(output_dir, result_filename)
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result

def process_video(
    model: YOLO,
    video_path: str,
    output_dir: str = "results",
    conf_threshold: float = 0.25,
    save_video: bool = True,
    display: bool = False
):
    """
    Process a video file for collision detection.
    
    Args:
        model: YOLOv8 model
        video_path: Path to input video
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for detections
        save_video: Whether to save the output video
        display: Whether to display the video (requires GUI support)
    
    Returns:
        Dictionary with detection results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
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
    output_path = None
    if save_video:
        output_filename = f"{Path(video_path).stem}_detected.mp4"
        output_path = os.path.join(output_dir, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize metrics
    metrics = {
        'input_path': video_path,
        'output_path': output_path,
        'total_frames': total_frames,
        'processed_frames': 0,
        'total_inference_time': 0,
        'fps': [],
        'detections_per_frame': [],
        'classes_detected': set()
    }
    
    # Check if display is available
    display_available = False
    if display:
        try:
            # Try to create a window to check if GUI is available
            cv2.namedWindow('YOLOv8 Collision Detection', cv2.WINDOW_NORMAL)
            cv2.destroyAllWindows()
            display_available = True
        except cv2.error:
            print("Warning: GUI display is not available. Video will be processed without display.")
    
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
        
        # Display frame if requested and GUI is available
        if display and display_available:
            try:
                cv2.imshow('YOLOv8 Collision Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except cv2.error:
                display_available = False
                print("Warning: GUI display failed. Continuing without display.")
        
        metrics['processed_frames'] += 1
        
        # Print progress every 30 frames
        if metrics['processed_frames'] % 30 == 0:
            print(f"Processed {metrics['processed_frames']}/{total_frames} frames")
    
    # Cleanup
    cap.release()
    if save_video:
        out.release()
    if display and display_available:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    
    # Calculate final metrics
    metrics['average_fps'] = np.mean(metrics['fps'])
    metrics['average_detections'] = np.mean(metrics['detections_per_frame'])
    metrics['classes_detected'] = list(metrics['classes_detected'])
    
    # Save results to JSON
    result_filename = f"{Path(video_path).stem}_results.json"
    result_path = os.path.join(output_dir, result_filename)
    with open(result_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Run YOLOv8 collision detection on images or videos')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained YOLOv8 model')
    parser.add_argument('--input_path', type=str, required=True,
                      help='Path to input image or video file')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                      help='Confidence threshold for detections')
    parser.add_argument('--no_save', action='store_true',
                      help='Do not save output files')
    parser.add_argument('--display', action='store_true',
                      help='Display results (requires GUI support)')
    
    args = parser.parse_args()
    
    # Initialize model
    print(f"Loading model from {args.model_path}...")
    model = YOLO(args.model_path)
    
    # Determine if input is image or video
    input_path = Path(args.input_path)
    print(f"Processing {input_path}...")
    
    if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        # Process image
        print("Detected image input. Running collision detection...")
        result = process_image(
            model=model,
            image_path=str(input_path),
            output_dir=args.output_dir,
            conf_threshold=args.conf_threshold,
            save_output=not args.no_save,
            display=args.display
        )
        print("\nImage Processing Results:")
    else:
        # Process video
        print("Detected video input. Running collision detection...")
        result = process_video(
            model=model,
            video_path=str(input_path),
            output_dir=args.output_dir,
            conf_threshold=args.conf_threshold,
            save_video=not args.no_save,
            display=args.display
        )
        print("\nVideo Processing Results:")
    
    # Print results
    print(json.dumps(result, indent=2))
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == '__main__':
    main() 