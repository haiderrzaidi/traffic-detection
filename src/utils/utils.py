import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_model(model: torch.nn.Module, save_path: str) -> None:
    """Save model weights to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

def load_model(model: torch.nn.Module, load_path: str) -> torch.nn.Module:
    """Load model weights from disk."""
    model.load_state_dict(torch.load(load_path))
    return model

def plot_training_history(
    history: Dict[str, List[float]],
    save_path: str = None
) -> None:
    """Plot training history."""
    plt.figure(figsize=(10, 5))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str = None
) -> None:
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def print_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str]
) -> None:
    """Print classification metrics."""
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

def calculate_model_size(model: torch.nn.Module) -> Tuple[int, float]:
    """Calculate model size in parameters and MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return sum(p.numel() for p in model.parameters()), size_all_mb

def visualize_predictions(
    frames: torch.Tensor,
    predictions: torch.Tensor,
    true_labels: torch.Tensor,
    class_names: List[str],
    num_samples: int = 5,
    save_path: str = None
) -> None:
    """Visualize model predictions on sample frames."""
    num_samples = min(num_samples, len(frames))
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 5*num_samples))
    
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        # Convert tensor to numpy and denormalize
        frame = frames[i].permute(1, 2, 0).cpu().numpy()
        frame = (frame * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
        frame = frame.astype(np.uint8)
        
        pred_label = class_names[predictions[i]]
        true_label = class_names[true_labels[i]]
        
        axes[i].imshow(frame)
        axes[i].set_title(f'Pred: {pred_label} | True: {true_label}')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close() 