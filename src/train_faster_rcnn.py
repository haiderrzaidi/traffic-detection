import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader, random_split
import json
import random
from pathlib import Path
import argparse
import time
from PIL import Image
import numpy as np
from tqdm import tqdm

class CarCollisionDataset(Dataset):
    def __init__(self, root_dir, transform=None, data_fraction=1.0):
        """
        Car Collision Dataset in COCO format
        
        Args:
            root_dir: Root directory of the dataset
            transform: Optional transform to be applied on a sample
            data_fraction: Fraction of dataset to use (0.0 to 1.0)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Load COCO annotations
        self.images_dir = self.root_dir / "train"
        self.annotations_file = self.root_dir / "train" / "_annotations.coco.json"
        
        if not self.annotations_file.exists():
            raise FileNotFoundError(f"COCO annotations file not found at {self.annotations_file}")
        
        with open(self.annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Get image IDs
        self.image_ids = list(self.coco_data['images'])
        
        # Apply data fraction
        if data_fraction < 1.0:
            num_samples = int(len(self.image_ids) * data_fraction)
            self.image_ids = random.sample(self.image_ids, num_samples)
        
        # Create image ID to annotations mapping
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        # Create category ID to index mapping
        self.cat_to_idx = {cat['id']: idx for idx, cat in enumerate(self.coco_data['categories'])}
        self.idx_to_cat = {idx: cat_id for cat_id, idx in self.cat_to_idx.items()}
        
        print(f"Loaded {len(self.image_ids)} images with {len(self.coco_data['annotations'])} annotations")
        print(f"Categories: {[cat['name'] for cat in self.coco_data['categories']]}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_info = self.image_ids[idx]
        img_id = img_info['id']
        
        # Load image
        img_path = self.images_dir / img_info['file_name']
        img = Image.open(img_path).convert("RGB")
        
        # Get annotations for this image
        anns = self.img_to_anns.get(img_id, [])
        
        # Prepare boxes and labels
        boxes = []
        labels = []
        
        for ann in anns:
            # COCO format: [x_min, y_min, width, height]
            bbox = ann['bbox']
            # Convert to [x_min, y_min, x_max, y_max]
            x_min, y_min, width, height = bbox
            boxes.append([x_min, y_min, x_min + width, y_min + height])
            
            # Get category index
            cat_id = ann['category_id']
            labels.append(self.cat_to_idx[cat_id])
        
        # Convert to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
        }
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        return img, target

def get_model(num_classes):
    """
    Get Faster R-CNN model with custom number of classes
    """
    # Load pre-trained model
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    
    # Replace the classifier with a new one for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def train_one_epoch(model, optimizer, data_loader, device):
    """
    Train the model for one epoch
    """
    model.train()
    
    total_loss = 0
    num_batches = len(data_loader)
    successful_batches = 0
    
    for images, targets in tqdm(data_loader, desc="Training"):
        try:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
            successful_batches += 1
        except Exception as e:
            print(f"Warning: Error during training batch: {e}")
            # Continue with next batch
            continue
    
    # Avoid division by zero
    if successful_batches == 0:
        return float('inf')
    
    return total_loss / successful_batches

def evaluate(model, data_loader, device):
    """
    Evaluate the model on the validation set
    """
    model.eval()
    
    total_loss = 0
    num_batches = len(data_loader)
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            try:
                loss_dict = model(images, targets)
                # Handle different loss dictionary formats
                if isinstance(loss_dict, dict):
                    losses = sum(loss for loss in loss_dict.values())
                else:
                    # If it's not a dictionary, assume it's a tensor
                    losses = loss_dict
                
                total_loss += losses.item()
            except Exception as e:
                print(f"Warning: Error during evaluation: {e}")
                # Continue with next batch
                continue
    
    # Avoid division by zero
    if num_batches == 0:
        return float('inf')
    
    return total_loss / num_batches

def main():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN on car collision dataset')
    parser.add_argument('--data_dir', type=str, default='data/raw/car_collision_coco',
                      help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='runs/faster_rcnn',
                      help='Directory to save model and logs')
    parser.add_argument('--epochs', type=int, default=1,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                      help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.005,
                      help='Learning rate')
    parser.add_argument('--data_fraction', type=float, default=0.08,
                      help='Fraction of dataset to use (0.0 to 1.0)')
    parser.add_argument('--val_split', type=float, default=0.2,
                      help='Fraction of data to use for validation (0.0 to 1.0)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Create dataset
        dataset = CarCollisionDataset(
            root_dir=args.data_dir,
            transform=torchvision.transforms.ToTensor(),
            data_fraction=args.data_fraction
        )
        
        # Split dataset into train and validation
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=lambda x: tuple(zip(*x))
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=lambda x: tuple(zip(*x))
        )
        
        # Create model
        num_classes = len(dataset.coco_data['categories']) + 1  # +1 for background
        model = get_model(num_classes)
        model.to(device)
        
        # Create optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
        
        # Training loop
        print(f"Starting training for {args.epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}")
            
            # Train
            train_loss = train_one_epoch(model, optimizer, train_loader, device)
            print(f"Training Loss: {train_loss:.4f}")
            
            # Save model after each epoch
            model_path = os.path.join(args.output_dir, f"faster_rcnn_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, model_path)
            print(f"Model saved to {model_path}")
            
            # Evaluate
            try:
                val_loss = evaluate(model, val_loader, device)
                print(f"Validation Loss: {val_loss:.4f}")
                
                # Update saved model with validation loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, model_path)
            except Exception as e:
                print(f"Warning: Evaluation failed: {e}")
                print("Continuing with training...")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save final model
        final_model_path = os.path.join(args.output_dir, "faster_rcnn_final.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'num_classes': num_classes,
            'cat_to_idx': dataset.cat_to_idx,
            'idx_to_cat': dataset.idx_to_cat,
        }, final_model_path)
        print(f"Final model saved to {final_model_path}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        # Try to save the model even if there was an error
        try:
            if 'model' in locals():
                emergency_save_path = os.path.join(args.output_dir, "faster_rcnn_emergency_save.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'num_classes': num_classes if 'num_classes' in locals() else None,
                    'cat_to_idx': dataset.cat_to_idx if 'dataset' in locals() else None,
                    'idx_to_cat': dataset.idx_to_cat if 'dataset' in locals() else None,
                }, emergency_save_path)
                print(f"Emergency model save to {emergency_save_path}")
        except Exception as save_error:
            print(f"Failed to save emergency model: {save_error}")

if __name__ == '__main__':
    main() 