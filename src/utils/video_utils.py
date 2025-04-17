import cv2
import numpy as np
import torch
from typing import Tuple, List
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

def preprocess_frame(frame: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """Preprocess a single frame for model input."""
    # Resize frame
    frame = cv2.resize(frame, target_size)
    
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    frame = frame.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
    
    return frame

def create_video_writer(output_path: str, frame_size: Tuple[int, int], fps: float = 30.0) -> cv2.VideoWriter:
    """Create a video writer for saving output."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Use H.264 codec for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer for {output_path}")
    
    return writer

def add_prediction_to_frame(frame: np.ndarray, prediction: float, threshold: float = 0.5) -> np.ndarray:
    """Add prediction text to the frame."""
    # Convert prediction to probability
    prob = torch.sigmoid(torch.tensor(prediction)).item()
    
    # Determine prediction text and color
    prediction_text = "Collision Likely" if prob > threshold else "No Collision"
    color = (0, 0, 255) if prob > threshold else (0, 255, 0)  # Red for collision, Green for no collision
    
    # Add text to frame
    cv2.putText(frame, f"Prediction: {prediction_text}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Probability: {prob:.2f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return frame

def process_video_frames(video_path, model, device, target_size=(224, 224)):
    """Process video frames and make predictions."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    results = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        processed_frame = preprocess_frame(frame, target_size)
        processed_frame = processed_frame.to(device)

        # Make prediction
        with torch.no_grad():
            output = model(processed_frame)
            print(f"Model output type: {type(output)}")
            print(f"Model output shape: {output.shape if hasattr(output, 'shape') else 'No shape'}")
            print(f"Model output: {output}")

            # Convert to probability using softmax for multi-class
            probabilities = torch.softmax(output, dim=1)
            # Get the probability of the positive class (index 1)
            prediction = probabilities[0, 1].item()

        results.append((frame, prediction))
        frame_count += 1

    cap.release()
    return results

def save_prediction_video(
    predictions: List[Tuple[np.ndarray, float]],
    output_path: str,
    threshold: float = 0.5
) -> None:
    """Save video with predictions overlaid."""
    if not predictions:
        print("No predictions to save")
        return
        
    print(f"Starting to save video to {output_path}")
    print(f"Number of frames to save: {len(predictions)}")
    
    # Get frame size from first frame
    frame_size = predictions[0][0].shape[1], predictions[0][0].shape[0]
    print(f"Frame size: {frame_size}")
    
    try:
        # Create video writer
        print("Creating video writer...")
        writer = create_video_writer(output_path, frame_size)
        
        # Process and write frames
        print("Writing frames to video...")
        for i, (frame, prediction) in enumerate(predictions):
            if i % 10 == 0:
                print(f"Writing frame {i+1}/{len(predictions)}")
            frame_with_pred = add_prediction_to_frame(frame, prediction, threshold)
            writer.write(frame_with_pred)
        
        writer.release()
        print(f"Successfully wrote {len(predictions)} frames to {output_path}")
    except Exception as e:
        print(f"Error saving video: {e}")
        raise

def save_prediction_frames(
    predictions: List[Tuple[np.ndarray, float]],
    output_dir: str,
    threshold: float = 0.5
) -> None:
    """Save individual frames with predictions."""
    if not predictions:
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each frame
    for i, (frame, prediction) in enumerate(predictions):
        frame_with_pred = add_prediction_to_frame(frame, prediction, threshold)
        output_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(output_path, frame_with_pred)

def visualize_predictions(
    predictions: List[Tuple[np.ndarray, float]],
    threshold: float = 0.5,
    output_dir: str = None
) -> None:
    """Create visualization of predictions."""
    if output_dir:
        # Save frames to directory
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving {len(predictions)} frames to {output_dir}")
        for i, (frame, prediction) in enumerate(predictions):
            # Add prediction text to frame
            frame_with_text = add_prediction_to_frame(frame, prediction, threshold)
            
            # Save frame
            output_path = os.path.join(output_dir, f'frame_{i:04d}.jpg')
            cv2.imwrite(output_path, frame_with_text)
            
            if i % 10 == 0:
                print(f"Saved frame {i}/{len(predictions)}")
        
        print(f"Successfully saved all frames to {output_dir}")
    else:
        # Try to show interactive visualization
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            def update(frame_idx):
                ax.clear()
                frame, prediction = predictions[frame_idx]
                frame_with_pred = add_prediction_to_frame(frame.copy(), prediction, threshold)
                ax.imshow(cv2.cvtColor(frame_with_pred, cv2.COLOR_BGR2RGB))
                ax.set_title(f'Frame {frame_idx + 1}/{len(predictions)}')
                ax.axis('off')
            
            anim = FuncAnimation(fig, update, frames=len(predictions), interval=100)
            plt.show()
        except Exception as e:
            print(f"Could not show interactive visualization: {e}")
            print("Saving frames to 'temp_frames' directory instead...")
            # Save to project root's temp_frames directory
            temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'temp_frames')
            os.makedirs(temp_dir, exist_ok=True)
            save_prediction_frames(predictions, temp_dir, threshold) 