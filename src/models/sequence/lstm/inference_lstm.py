import os
import sys
import torch
import argparse
from pathlib import Path

# Get the absolute path of the project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# Add src to Python path
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from models.lstm.lstm_model import LSTMModel
from utils.video_utils import process_video_frames, save_prediction_video, visualize_predictions

def load_model(checkpoint_path: str) -> LSTMModel:
    """Load the trained LSTM model from checkpoint."""
    # Convert checkpoint path to absolute path if it's relative
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(project_root, checkpoint_path)
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        # Try alternative path in checkpoints directory
        alt_path = os.path.join(project_root, 'checkpoints', f'lstm-{os.path.basename(checkpoint_path)}')
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
        else:
            # List available checkpoints
            checkpoints_dir = os.path.join(project_root, 'checkpoints')
            if os.path.exists(checkpoints_dir):
                print("Available checkpoints:")
                for file in os.listdir(checkpoints_dir):
                    if file.startswith('lstm-'):
                        print(f"- {file}")
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    model = LSTMModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description='Run inference on a video using LSTM model')
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_path', type=str, default='output_lstm.mp4', help='Path to save output video')
    parser.add_argument('--sequence_length', type=int, default=16, help='Number of frames in each sequence')
    parser.add_argument('--visualize', action='store_true', help='Show visualization of predictions')
    parser.add_argument('--frames_dir', type=str, help='Directory to save individual frames')
    args = parser.parse_args()

    # Convert relative paths to absolute
    video_path = os.path.join(project_root, args.video_path) if not os.path.isabs(args.video_path) else args.video_path
    checkpoint_path = os.path.join(project_root, args.checkpoint_path) if not os.path.isabs(args.checkpoint_path) else args.checkpoint_path
    output_path = os.path.join(project_root, args.output_path) if not os.path.isabs(args.output_path) else args.output_path

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load model
    model = load_model(checkpoint_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)

    # Process video
    print("Processing video frames...")
    predictions = process_video_frames(video_path, model, device, sequence_length=args.sequence_length)

    # Save output video
    print(f"Saving output video to {output_path}...")
    save_prediction_video(predictions, output_path)

    # Handle visualization
    if args.visualize or args.frames_dir:
        print("Processing visualization...")
        # Use project root for frames directory
        frames_dir = os.path.join(project_root, 'temp_frames_lstm')
        if args.frames_dir:
            frames_dir = os.path.join(project_root, args.frames_dir)
        
        # Create frames directory
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Saving frames to: {frames_dir}")
        
        # Save frames
        visualize_predictions(predictions, output_dir=frames_dir)
        print(f"Successfully saved frames to: {frames_dir}")

if __name__ == '__main__':
    main() 