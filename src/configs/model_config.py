"""Configuration file for model parameters and hyperparameters."""

# Common training parameters
TRAINING_CONFIG = {
    'batch_size': 32,
    'num_workers': 4,
    'max_epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'early_stopping_patience': 10,
}

# Object Detection Models Configurations
YOLO_CONFIG = {
    'model_name': 'yolov7',
    'pretrained': True,
    'confidence_threshold': 0.5,
    'nms_threshold': 0.45,
    'input_size': (640, 640),
}

FASTER_RCNN_CONFIG = {
    'backbone': 'resnet50',
    'pretrained_backbone': True,
    'num_classes': 2,  # background + accident
    'min_size': 800,
    'max_size': 1333,
}

SSD_CONFIG = {
    'backbone': 'vgg16',
    'pretrained_backbone': True,
    'num_classes': 2,
    'min_size': 300,
}

DETR_CONFIG = {
    'num_classes': 2,
    'hidden_dim': 256,
    'nheads': 8,
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
}

EFFICIENTDET_CONFIG = {
    'compound_coef': 0,  # D0 variant
    'num_classes': 2,
}

# Classification Models Configurations
RESNET_CONFIG = {
    'model_name': 'resnet50',
    'pretrained': True,
    'num_classes': 2,  # binary classification: no accident (0) or accident (1)
}

EFFICIENTNET_CONFIG = {
    'model_name': 'efficientnet_b0',
    'pretrained': True,
    'num_classes': 3,
}

VIT_CONFIG = {
    'model_name': 'vit_base_patch16_224',
    'pretrained': True,
    'num_classes': 3,
}

# Sequence Models Configurations
LSTM_CONFIG = {
    'input_size': 2048,  # Feature vector size from CNN backbone
    'hidden_size': 512,
    'num_layers': 2,
    'dropout': 0.5,
    'bidirectional': True,
    'sequence_length': 16,  # Number of frames to process
}

GRU_CONFIG = {
    'input_size': 2048,
    'hidden_size': 512,
    'num_layers': 2,
    'dropout': 0.5,
    'bidirectional': True,
    'sequence_length': 16,
}

# Dataset Configuration
DATASET_CONFIG = {
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'sequence_length': 16,  # Number of frames to sample from each video
    'image_size': (224, 224),  # Input image size for classification models
    'batch_size': 8,  # Reduced from 32 due to memory constraints with video data
    'num_workers': 4,
} 