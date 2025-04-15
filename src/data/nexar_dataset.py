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

from src.configs.model_config import DATASET_CONFIG

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
        
        # Filter out videos that don't exist
        available_videos = set(int(f.split('.')[0]) for f in os.listdir(root_dir) if f.endswith('.mp4'))
        self.df = df[df['id'].astype(int).isin(available_videos)].reset_index(drop=True)
        print(f"Found {len(self.df)} videos out of {len(df)} annotations")
        
        self.transform = transform
        self.frame_count = DATASET_CONFIG['sequence_length']

    def __len__(self) -> int:
        return len(self.df)

    def _extract_frames(self, video_path: str) -> np.ndarray:
        """Extract frames from video file."""
        if not os.path.exists(video_path):
            raise RuntimeError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
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
        video_id = str(int(row['id'])).zfill(5)  # Convert to int first to remove decimal, then pad with zeros
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
            A.Resize(
                height=DATASET_CONFIG['image_size'][0],
                width=DATASET_CONFIG['image_size'][1]
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def setup(self, stage: Optional[str] = None):
        """Load data and split into train, validation, and test sets."""
        # Read annotations
        train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        
        # Split data
        train_size = int(len(train_df) * DATASET_CONFIG['train_split'])
        val_size = int(len(train_df) * DATASET_CONFIG['val_split'])

        train_df = train_df[:train_size]
        val_df = train_df[train_size:train_size + val_size]

        if stage == 'fit' or stage is None:
            self.train_dataset = NexarDataset(
                os.path.join(self.data_dir, 'train'),
                train_df,
                transform=self.train_transform
            )
            self.val_dataset = NexarDataset(
                os.path.join(self.data_dir, 'train'),
                val_df,
                transform=self.val_transform
            )

        if stage == 'test' or stage is None:
            test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))
            self.test_dataset = NexarDataset(
                os.path.join(self.data_dir, 'test'),
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