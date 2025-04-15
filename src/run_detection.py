import os
import argparse
from src.pipeline.accident_detection_pipeline import AccidentDetectionPipeline

def main():
    parser = argparse.ArgumentParser(description='Run real-time accident detection')
    parser.add_argument('--video', type=str, default='0',
                       help='Path to video file or camera index (default: 0 for webcam)')
    parser.add_argument('--yolo-checkpoint', type=str,
                       help='Path to YOLOv7 checkpoint')
    parser.add_argument('--efficientdet-checkpoint', type=str,
                       help='Path to EfficientDet checkpoint')
    parser.add_argument('--lstm-checkpoint', type=str,
                       help='Path to LSTM checkpoint')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold')
    args = parser.parse_args()

    # Initialize pipeline
    pipeline = AccidentDetectionPipeline(
        yolo_checkpoint=args.yolo_checkpoint,
        efficientdet_checkpoint=args.efficientdet_checkpoint,
        lstm_checkpoint=args.lstm_checkpoint,
        confidence_threshold=args.confidence
    )

    # Process video stream
    video_source = int(args.video) if args.video.isdigit() else args.video
    pipeline.process_video_stream(video_source)

if __name__ == '__main__':
    main() 