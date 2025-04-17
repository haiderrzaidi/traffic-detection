import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
import argparse
from pathlib import Path
import time
import json
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def load_model(model_path, device):
    """
    Load the trained Faster R-CNN model
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get number of classes
    num_classes = checkpoint['num_classes']
    
    # Create model
    model = fasterrcnn_resnet50_fpn_v2(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Get category mappings
    cat_to_idx = checkpoint['cat_to_idx']
    idx_to_cat = checkpoint['idx_to_cat']
    
    return model, cat_to_idx, idx_to_cat

def process_image(model, image_path, output_dir, conf_threshold, device, cat_to_idx, idx_to_cat, save_output=True, display=False):
    """
    Process a single image with the Faster R-CNN model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_tensor = torchvision.transforms.ToTensor()(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Run inference
    start_time = time.time()
    with torch.no_grad():
        predictions = model(image_tensor)
    inference_time = time.time() - start_time
    
    # Process predictions
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    
    # Filter by confidence threshold
    mask = scores >= conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    # Convert to numpy image for OpenCV
    img_np = np.array(image)
    
    # Draw detections
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)
        label_name = list(cat_to_idx.keys())[list(cat_to_idx.values()).index(label)]
        
        # Draw rectangle
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label_text = f"{label_name}: {score:.2f}"
        cv2.putText(img_np, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save output if requested
    output_path = None
    if save_output:
        output_filename = f"{Path(image_path).stem}_detected{Path(image_path).suffix}"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    
    # Display if requested
    if display:
        plt.figure(figsize=(12, 8))
        plt.imshow(img_np)
        plt.axis('off')
        plt.show()
    
    # Prepare results
    result = {
        'input_path': str(image_path),
        'output_path': output_path,
        'inference_time': inference_time,
        'num_detections': len(boxes),
        'detections': [
            {
                'box': box.tolist(),
                'score': float(score),
                'label': int(label),
                'label_name': list(cat_to_idx.keys())[list(cat_to_idx.values()).index(label)]
            }
            for box, score, label in zip(boxes, scores, labels)
        ]
    }
    
    # Save results to JSON
    result_filename = f"{Path(image_path).stem}_results.json"
    result_path = os.path.join(output_dir, result_filename)
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result

def process_video(model, video_path, output_dir, conf_threshold, device, cat_to_idx, idx_to_cat, save_video=True, display=False):
    """
    Process a video file with the Faster R-CNN model
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
        'input_path': str(video_path),
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
            cv2.namedWindow('Faster R-CNN Detection', cv2.WINDOW_NORMAL)
            cv2.destroyAllWindows()
            display_available = True
        except cv2.error:
            print("Warning: GUI display is not available. Video will be processed without display.")
    
    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to RGB for model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image_tensor = torchvision.transforms.ToTensor()(image)
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Run inference
        start_time = time.time()
        with torch.no_grad():
            predictions = model(image_tensor)
        inference_time = time.time() - start_time
        
        # Process predictions
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        # Filter by confidence threshold
        mask = scores >= conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        # Update metrics
        metrics['total_inference_time'] += inference_time
        metrics['fps'].append(1.0 / inference_time if inference_time > 0 else 0)
        metrics['detections_per_frame'].append(len(boxes))
        metrics['classes_detected'].update(set(labels))
        
        # Draw detections
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.astype(int)
            label_name = list(cat_to_idx.keys())[list(cat_to_idx.values()).index(label)]
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label_text = f"{label_name}: {score:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save frame if needed
        if save_video:
            out.write(frame)
        
        # Display frame if requested and GUI is available
        if display and display_available:
            try:
                cv2.imshow('Faster R-CNN Detection', frame)
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
    parser = argparse.ArgumentParser(description='Run Faster R-CNN inference on images or videos')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained Faster R-CNN model')
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
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model, cat_to_idx, idx_to_cat = load_model(args.model_path, device)
    
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
            device=device,
            cat_to_idx=cat_to_idx,
            idx_to_cat=idx_to_cat,
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
            device=device,
            cat_to_idx=cat_to_idx,
            idx_to_cat=idx_to_cat,
            save_video=not args.no_save,
            display=args.display
        )
        print("\nVideo Processing Results:")
    
    # Print results
    print(json.dumps(result, indent=2))
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == '__main__':
    main() 