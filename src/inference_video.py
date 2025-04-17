import cv2
import torch
from ultralytics import YOLO
import argparse
from pathlib import Path

def process_video(
    model_path: str,
    video_path: str,
    output_path: str = None,
    conf_threshold: float = 0.25,
    save_video: bool = True
):
    """
    Process a video file using YOLOv8 model for car collision detection.
    
    Args:
        model_path: Path to the trained YOLOv8 model
        video_path: Path to input video file
        output_path: Path to save the output video (if None, will use input name with _output suffix)
        conf_threshold: Confidence threshold for detections
        save_video: Whether to save the output video
    """
    # Initialize YOLO model
    model = YOLO(model_path)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Setup output video writer if needed
    if save_video:
        if output_path is None:
            output_path = str(Path(video_path).with_name(Path(video_path).stem + '_output.mp4'))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run inference
        results = model(frame, conf=conf_threshold)[0]
        
        # Draw detections on frame
        annotated_frame = results.plot()
        
        # Save frame if needed
        if save_video:
            out.write(annotated_frame)
        
        # Display frame
        cv2.imshow('YOLOv8 Inference', annotated_frame)
        
        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_count += 1
        if frame_count % 30 == 0:  # Print progress every 30 frames
            print(f"Processed {frame_count} frames")
    
    # Cleanup
    cap.release()
    if save_video:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"\nProcessing complete!")
    if save_video:
        print(f"Output video saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Run YOLOv8 inference on video')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained YOLOv8 model')
    parser.add_argument('--video_path', type=str, required=True,
                      help='Path to input video file')
    parser.add_argument('--output_path', type=str, default=None,
                      help='Path to save output video (optional)')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                      help='Confidence threshold for detections')
    parser.add_argument('--no_save', action='store_true',
                      help='Do not save output video')
    
    args = parser.parse_args()
    
    process_video(
        model_path=args.model_path,
        video_path=args.video_path,
        output_path=args.output_path,
        conf_threshold=args.conf_threshold,
        save_video=not args.no_save
    )

if __name__ == '__main__':
    main() 