import torch
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
import os
import xml.etree.ElementTree as ET
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_class_names(annotation_dir):
    """
    Scan XML annotations to extract unique class names.
    Returns a set of class names.
    """
    try:
        class_names = set()
        if not os.path.exists(annotation_dir):
            logger.error(f"Annotation directory {annotation_dir} does not exist")
            return class_names

        for xml_file in os.listdir(annotation_dir):
            if not xml_file.endswith('.xml'):
                continue
            xml_path = os.path.join(annotation_dir, xml_file)
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    class_name = obj.find('name').text.lower()
                    class_names.add(class_name)
            except ET.ParseError as e:
                logger.warning(f"Error parsing {xml_file}: {str(e)}")
        logger.info(f"Found classes: {class_names}")
        return class_names
    except Exception as e:
        logger.error(f"Error scanning annotations: {str(e)}")
        return set()

def get_efficientdet_model(data_dir='./dataset', model_name='tf_efficientdet_d0'):
    """
    Initialize EfficientDet model with dynamically determined number of classes.
    """
    try:
        # Get class names from training annotations
        annotation_dir = os.path.join(data_dir, 'train', 'annotations')
        class_names = get_class_names(annotation_dir)
        
        if not class_names:
            logger.error("No classes found in annotations. Using default num_classes=5.")
            num_classes = 5
        else:
            # Assume 'null' is a regular class, not background
            num_classes = len(class_names)
            logger.info(f"Number of classes (including all): {num_classes}")

        # Get model configuration
        config = get_efficientdet_config(model_name)
        config.num_classes = num_classes
        config.image_size = [256, 256]  # Match dataloader
        # Initialize model
        model = EfficientDet(config, pretrained_backbone=True)
        # Wrap for training
        model = DetBenchTrain(model, config)
        return model
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise

if __name__ == '__main__':
    model = get_efficientdet_model()
    print(model)