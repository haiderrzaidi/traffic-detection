import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VOCDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=256):
        self.root_dir = os.path.join(root_dir, split)
        self.img_size = img_size
        self.image_dir = os.path.join(self.root_dir, 'images')
        self.annotation_dir = os.path.join(self.root_dir, 'annotations')
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        # Class mapping (adjust if class names differ)
        self.class_to_id = {'null': 0, 'car': 1, 'truck': 2, 'person': 3, 'bicycle': 4}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            # Load image
            img_path = os.path.join(self.image_dir, self.image_files[idx])
            img = Image.open(img_path).convert('RGB')  # Convert grayscale to RGB
            img = self.transform(img)

            # Load annotation
            annotation_path = os.path.join(self.annotation_dir, self.image_files[idx].replace('.jpg', '.xml'))
            boxes, labels = self.parse_voc_xml(annotation_path)

            # Convert to tensors
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            target = {'boxes': boxes, 'labels': labels}

            return img, target
        except Exception as e:
            logger.error(f"Error processing {self.image_files[idx]}: {str(e)}")
            return None

    def parse_voc_xml(self, xml_file):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            boxes = []
            labels = []
            for obj in root.findall('object'):
                class_name = obj.find('name').text.lower()
                if class_name not in self.class_to_id:
                    logger.warning(f"Unknown class {class_name} in {xml_file}")
                    continue
                label_id = self.class_to_id[class_name]
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label_id)
            if not boxes:  # Handle empty annotations
                boxes = [[0, 0, 0, 0]]
                labels = [0]  # Null class for background
            return boxes, labels
        except ET.ParseError as e:
            logger.error(f"XML parsing error in {xml_file}: {str(e)}")
            return [[0, 0, 0, 0]], [0]

def get_dataloader(root_dir, split='train', batch_size=16, num_workers=4):
    dataset = VOCDataset(root_dir, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'),
                            num_workers=num_workers, collate_fn=collate_fn)
    return dataloader

def collate_fn(batch):
    batch = [b for b in batch if b is not None]  # Filter out None entries
    if not batch:
        return torch.tensor([]), []
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets