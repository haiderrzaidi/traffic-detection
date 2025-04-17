import os
from roboflow import Roboflow

def download_dataset(api_key: str, output_dir: str = "data/raw/car-collision"):
    """
    Download the car collision dataset from Roboflow.
    
    Args:
        api_key: Roboflow API key
        output_dir: Directory to save the dataset
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Roboflow
    rf = Roboflow(api_key="9pUh2QbWnXHuh55IIHGU")
    project = rf.workspace("t5-xhhs7").project("car-collision_system")
    version = project.version(3)

    
    # Download dataset in YOLOv8 format
    print("Downloading dataset...")
    dataset = version.download("yolov8", output_dir)
    print(f"Dataset downloaded to: {output_dir}")
    
    return dataset

if __name__ == "__main__":
    # Replace with your API key
    API_KEY = "9pUh2QbWnXHuh55IIHGU"
    download_dataset(API_KEY) 
                