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

## Setup Instructions

1. Clone the repository:
```bash
git clone [repository-url]
cd traffic_incident_detection
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
- Visit https://www.kaggle.com/competitions/nexar-collision-prediction/data
- Download and extract the dataset to `src/data/raw/`

## Usage
[Usage instructions will be added as the project develops]

## Project Timeline
- Phase 1 (Mar 17 - Mar 23): Research and dataset review
- Phase 2 (Mar 24 - Mar 30): System architecture design
- Phase 3 (Mar 31 - Apr 5): Model implementation and training
- Phase 4 (Apr 6 - Apr 12): Benchmarking and system refinement
- Phase 5 (Apr 13 - Apr 20): Documentation and project submission

## Contributing
[Contribution guidelines will be added]

## License
[License information will be added] 