#         sample_size = max(1, min(len(self.df), int(len(self.df) * 0.12))) just change 0.12 to 1.0
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2

from configs.model_config import DATASET_CONFIG

class NexarDataset(Dataset):
    """Dataset class for Nexar Collision Prediction Dataset."""
    
    def __init__(self, root_dir: str, df: pd.DataFrame, transform: Optional[A.Compose] = None):
        """
        Args:
            root_dir (str): Directory with all the video files
            df (pd.DataFrame): DataFrame containing the annotations
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        
        # Ensure the root directory exists
        if not os.path.exists(root_dir):
            raise ValueError(f"Root directory does not exist: {root_dir}")
        
        # Get available videos
        available_videos = set()
        for f in os.listdir(root_dir):
            if f.endswith('.mp4'):
                try:
                    # Remove .mp4 and convert to int to match CSV format
                    video_id = int(f.split('.')[0])
                    available_videos.add(video_id)
                except ValueError:
                    print(f"Warning: Skipping invalid video file name: {f}")
        
        if not available_videos:
            raise ValueError(f"No valid video files found in directory: {root_dir}")
        
        # Convert DataFrame IDs to int for comparison
        df['id'] = df['id'].astype(int)
        
        # Print some debug information
        print(f"First few video IDs in directory: {sorted(list(available_videos))[:10]}")
        print(f"First few IDs in DataFrame: {df['id'].head(10).tolist()}")
        
        # Create a mapping between video IDs and their padded versions
        id_mapping = {int(str(vid).zfill(5)): vid for vid in available_videos}
        
        # Filter DataFrame to only include available videos using the mapping
        self.df = df[df['id'].isin(id_mapping.keys())].reset_index(drop=True)
        
        if len(self.df) == 0:
            print(f"Available video IDs: {sorted(list(available_videos))[:10]}...")
            print(f"DataFrame IDs: {df['id'].head(10).tolist()}...")
            raise ValueError(f"No matching videos found between DataFrame and directory: {root_dir}")
        
        # Use only 5% of the data, but ensure we have at least 1 sample
        sample_size = max(1, min(len(self.df), int(len(self.df) * 0.12)))
        if sample_size > 0:
            self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        print(f"Using {len(self.df)} videos (10% of available data)")
        print(f"Available videos in directory: {len(available_videos)}")
        
        self.transform = transform
        self.frame_count = DATASET_CONFIG['sequence_length']
        self.id_mapping = id_mapping

    def __len__(self) -> int:
        return len(self.df)

    def _extract_frames(self, video_path: str) -> np.ndarray:
        """Extract frames from video file."""
        if not os.path.exists(video_path):
            raise RuntimeError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")
        
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            raise RuntimeError(f"Empty video file: {video_path}")
        
        # Calculate frame indices to sample
        frame_indices = np.linspace(0, total_frames-1, self.frame_count, dtype=int)
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in frame_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize frame to the target size
                frame = cv2.resize(frame, (DATASET_CONFIG['image_size'][1], DATASET_CONFIG['image_size'][0]))
                frames.append(frame)
        
        cap.release()
        
        if len(frames) < self.frame_count:
            raise RuntimeError(f"Could not extract enough frames from {video_path}")
        
        return np.array(frames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get sample information
        row = self.df.iloc[idx]
        video_id = str(self.id_mapping[row['id']]).zfill(5)  # Map to actual video ID and pad with zeros
        video_path = os.path.join(self.root_dir, f"{video_id}.mp4")
        
        # Extract frames from video
        frames = self._extract_frames(video_path)
        
        # Apply transforms if specified
        if self.transform:
            transformed_frames = []
            for frame in frames:
                transformed = self.transform(image=frame)['image']
                transformed_frames.append(transformed)
            frames = torch.stack(transformed_frames)
        else:
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # Convert to (T, C, H, W)
        
        # Get label
        label = torch.tensor(row['target'], dtype=torch.long)

        return frames, label

class NexarDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Nexar dataset."""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = DATASET_CONFIG['batch_size'],
        num_workers: int = DATASET_CONFIG['num_workers']
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Define transforms
        self.train_transform = A.Compose([
            A.Resize(height=DATASET_CONFIG['image_size'][0], width=DATASET_CONFIG['image_size'][1]),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        self.val_transform = A.Compose([
            A.Resize(height=DATASET_CONFIG['image_size'][0], width=DATASET_CONFIG['image_size'][1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def setup(self, stage: Optional[str] = None):
        """Load data and split into train, validation, and test sets."""
        # Read annotations
        train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        
        # Print debug information about the loaded data
        print(f"Total rows in train.csv: {len(train_df)}")
        print(f"First few rows of train.csv:")
        print(train_df.head())
        
        # Get the base directory for videos
        video_base_dir = os.path.join(self.data_dir, 'train')
        print(f"Looking for videos in: {video_base_dir}")

        # Create a single dataset first to get the filtered DataFrame
        full_dataset = NexarDataset(
            video_base_dir,
            train_df,
            transform=None
        )
        
        # Get the filtered DataFrame that matches available videos
        filtered_df = full_dataset.df
        
        # Split the filtered DataFrame
        train_size = int(len(filtered_df) * DATASET_CONFIG['train_split'])
        val_size = int(len(filtered_df) * DATASET_CONFIG['val_split'])
        
        train_df = filtered_df[:train_size]
        val_df = filtered_df[train_size:train_size + val_size]
        test_df = filtered_df[train_size + val_size:]
        
        print(f"Train set size: {len(train_df)}")
        print(f"Validation set size: {len(val_df)}")
        print(f"Test set size: {len(test_df)}")

        if stage == 'fit' or stage is None:
            self.train_dataset = NexarDataset(
                video_base_dir,
                train_df,
                transform=self.train_transform
            )
            self.val_dataset = NexarDataset(
                video_base_dir,
                val_df,
                transform=self.val_transform
            )

        if stage == 'test' or stage is None:
            self.test_dataset = NexarDataset(
                video_base_dir,
                test_df,
                transform=self.val_transform
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        ) 