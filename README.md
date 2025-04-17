# Automated Dashcam-Based Traffic Incident Detection

This project implements an automated system for detecting and classifying traffic incidents using dashcam footage. The system utilizes state-of-the-art deep learning models to provide real-time accident detection and severity classification.

## Project Structure
```
traffic_incident_detection/
├── src/
│   ├── models/         # Model implementations
│   ├── data/          # Data loading and preprocessing
│   ├── utils/         # Utility functions
│   └── configs/       # Configuration files
├── notebooks/         # Jupyter notebooks for experimentation
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Models Implemented
1. Object Detection Models:
   - YOLOv7
   - Faster R-CNN
   - SSD (Single Shot MultiBox Detector)
   - DETR (DEtection TRansformer)
   - EfficientDet

2. Image Classification Models:
   - ResNet-50
   - EfficientNet-B0
   - Vision Transformer (ViT)

3. Sequence-Based Models:
   - LSTM
   - GRU

## Dataset
This project uses the Nexar Collision Prediction Dataset from Kaggle. The dataset contains real-world dashcam footage with accident occurrence labels, GPS coordinates, vehicle speed, and timestamps.

## Dataset Configuration

The project uses the Nexar Collision Prediction Dataset with configurable sampling options:

### Dataset Sampling
By default, the system uses 5% of the available dataset to reduce computational requirements during development and testing. This can be modified in `src/data/nexar_dataset.py`:

```python
# To use 100% of the dataset, modify the __init__ method in NexarDataset:
def __init__(self, root_dir: str, df: pd.DataFrame, transform: Optional[A.Compose] = None):
    self.root_dir = root_dir
    
    # Filter out videos that don't exist
    available_videos = set(int(f.split('.')[0]) for f in os.listdir(root_dir) if f.endswith('.mp4'))
    self.df = df[df['id'].astype(int).isin(available_videos)].reset_index(drop=True)
    
    # Remove or modify these lines to use full dataset
    # sample_size = int(len(self.df) * 0.05)  # Comment out to use full dataset
    # self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    print(f"Using {len(self.df)} videos")
    
    self.transform = transform
    self.frame_count = DATASET_CONFIG['sequence_length']
```

### Dataset Splits
The dataset is split into training, validation, and test sets. The split ratios can be configured in `src/configs/model_config.py`:

```python
DATASET_CONFIG = {
    # ... other configurations ...
    'train_split': 0.7,  # 70% for training
    'val_split': 0.15,   # 15% for validation
    'test_split': 0.15,  # 15% for testing
    # ... other configurations ...
}
```

## Setup Instructions

1. Clone the repository:
```

## Model Inference

### GRU Model
The GRU model is a sequence-based model that processes multiple frames to detect traffic incidents.

```bash
python src/inference/gru_inference.py \
    --input_video /path/to/video.mp4 \
    --weights_path src/checkpoints/gru/gru-epoch=00-val_loss=0.69.ckpt \
    --output_dir output/
```

### EfficientNet Model
The EfficientNet model is a frame-based model that can classify traffic incidents into three categories: No Accident, Minor Accident, and Major Accident.

```bash
python src/inference/efficientnet_inference.py \
    --input_video /path/to/video.mp4 \
    --weights_path src/checkpoints/efficientnet/efficientnet-epoch=00-val_loss=0.81.ckpt \
    --output_dir output/
```

### Inference Output
Both models will:
1. Process the input video
2. Generate predictions:
   - GRU: Binary classification (No Accident/Accident)
   - EfficientNet: Multi-class classification (No Accident/Minor Accident/Major Accident)
3. Save a visualization with the prediction and confidence score
4. Print the results to the console

Output files:
- GRU: `output/gru_prediction.jpg`
- EfficientNet: `output/efficientnet_prediction.jpg`

### Example Usage
```bash
# Run GRU inference
python src/inference/gru_inference.py \
    --input_video src/data/raw/nexar-collision-prediction/test/00009.mp4 \
    --weights_path src/checkpoints/gru/gru-epoch=00-val_loss=0.69.ckpt \
    --output_dir output/

# Run EfficientNet inference
python src/inference/efficientnet_inference.py \
    --input_video src/data/raw/nexar-collision-prediction/test/00009.mp4 \
    --weights_path src/checkpoints/efficientnet/efficientnet-epoch=00-val_loss=0.81.ckpt \
    --output_dir output/
```

# Traffic Incident Detection

This project implements various deep learning models for traffic incident detection using video data.

## Models

The project includes the following models:

1. **YOLO (You Only Look Once)**

 **Training**

```bash
# Train YOLOv8 model on car collision dataset with 30% of data
python src/train_yolo_collision.py --data_yaml data/raw/car-collision/data.yaml --weights_path yolov8n.pt --output_dir runs/train --data_fraction 0.3

# Train with custom parameters
python src/train_yolo_collision.py --data_yaml data/raw/car-collision/data.yaml --weights_path yolov8n.pt --output_dir runs/train --epochs 100 --batch_size 16 --img_size 640 --data_fraction 0.3
```

## Inference

```bash
# Run inference on an image
python src/yolo_inference_collision.py --model_path runs/train/yolo_collision_training/weights/best.pt --input_path path/to/image.jpg

# Run inference on a video
python src/yolo_inference_collision.py --model_path runs/train/yolo_collision_training/weights/best.pt --input_path path/to/video.mp4

# Run inference with custom parameters
python src/yolo_inference_collision.py --model_path runs/train/yolo_collision_training/weights/best.pt --input_path path/to/input --output_dir custom_results --conf_threshold 0.3

# Run inference without saving output (just display)
python src/yolo_inference_collision.py --model_path runs/train/yolo_collision_training/weights/best.pt --input_path path/to/input --no_save

# Run inference with display (requires GUI support)
python src/yolo_inference_collision.py --model_path runs/train/yolo_collision_training/weights/best.pt --input_path path/to/input --display
```

## Parameters

### Training Parameters
- `--data_yaml`: Path to dataset YAML file (required)
- `--weights_path`: Path to pre-trained weights (default: yolov8n.pt)
- `--output_dir`: Directory to save training outputs (default: runs/train)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 16)
- `--img_size`: Input image size (default: 640)
- `--data_fraction`: Fraction of dataset to use (default: 0.3)

### Inference Parameters
- `--model_path`: Path to trained YOLOv8 model (required)
- `--input_path`: Path to input image or video file (required)
- `--output_dir`: Directory to save results (default: results)
- `--conf_threshold`: Confidence threshold for detections (default: 0.25)
- `--no_save`: Do not save output files
- `--display`: Display results (requires GUI support)

2. **GRU (Gated Recurrent Unit)**
   - Sequence model for temporal analysis of video frames
   - Training command:
     ```bash
     python src/models/sequence/gru/train_gru.py
     ```
   - Inference command:
     ```bash
     python src/inference/gru_inference.py --checkpoint_path /path/to/checkpoint.ckpt --input_video /path/to/video.mp4 --output_dir /path/to/save/results
     ```

3. **EfficientNet**
   - Efficient convolutional neural network for image classification
   - Training command:
     ```bash
     python src/models/classification/efficientnet/train_efficientnet.py
     ```
   - Inference command:
     ```bash
     python src/inference/efficientnet_inference.py --checkpoint_path /path/to/checkpoint.ckpt --input_video /path/to/video.mp4 --output_dir /path/to/save/results
     ```

4. **Vision Transformer (ViT)**
   - Transformer-based model for image classification
   - Training command:
     ```bash
     python src/models/classification/vit/train_vit.py
     ```
   - Inference command:
     ```bash
     python src/inference/vit_inference.py --checkpoint_path /path/to/checkpoint.ckpt --input_video /path/to/video.mp4 --output_dir /path/to/save/results
     ```

5. **ResNet and LSTM**
   - Combined model using ResNet for feature extraction and LSTM for sequence modeling
   - Training command:
     ```bash
     python src/train_all.py
     ```
   - This will train both ResNet and LSTM models

## Dataset

The project uses the Nexar Collision Prediction dataset. The dataset should be organized in the following structure:

```
data/
└── raw/
    └── nexar-collision-prediction/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── val/
        │   ├── images/
        │   └── labels/
        └── test/
            ├── images/
            └── labels/
```

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset according to the structure above
2. Choose the model you want to train
3. Run the appropriate training command
4. Use the trained model for inference on new videos

## Model Configurations

Model configurations can be found in `src/configs/model_config.py`. You can modify parameters such as:
- Learning rate
- Batch size
- Number of epochs
- Model architecture
- Training hyperparameters

## Output

- Trained models are saved in the `checkpoints` directory
- Inference results are saved in the specified output directory
- Training logs are saved in the `logs` directory