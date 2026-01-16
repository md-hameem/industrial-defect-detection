"""
Visualization utilities for anomaly detection results.

Includes:
- Reconstruction comparisons
- Anomaly heatmaps
- ROC curves
- Training curves
- Confusion matrices
- Dataset visualizations
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix


def set_style():
    """Set consistent plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def denormalize_image(img: torch.Tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) -> np.ndarray:
    """Denormalize image tensor for display."""
    if isinstance(img, torch.Tensor):
        img = img.cpu()
        if img.dim() == 4:
            img = img[0]
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)
        img = img * std + mean
        img = img.permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)


def plot_reconstruction_grid(
    originals: List[torch.Tensor],
    reconstructions: List[torch.Tensor],
    labels: List[int],
    error_maps: Optional[List[torch.Tensor]] = None,
    masks: Optional[List[torch.Tensor]] = None,
    title: str = "Reconstruction Results",
    save_path: Optional[Path] = None,
):
    """
    Plot grid of original, reconstruction, error map, and ground truth.
    
    Args:
        originals: List of original images
        reconstructions: List of reconstructed images
        labels: List of labels (0=normal, 1=anomaly)
        error_maps: Optional list of error maps
        masks: Optional list of ground truth masks
        title: Plot title
        save_path: Path to save figure
    """
    n_samples = len(originals)
    n_rows = 4 if masks is not None else (3 if error_maps is not None else 2)
    
    fig, axes = plt.subplots(n_rows, n_samples, figsize=(3 * n_samples, 3 * n_rows))
    if n_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_samples):
        # Original
        img_np = denormalize_image(originals[i])
        axes[0, i].imshow(img_np)
        label_str = "Defect" if labels[i] else "Normal"
        axes[0, i].set_title(f'{label_str}', fontsize=10)
        axes[0, i].axis('off')
        
        # Reconstruction
        recon_np = reconstructions[i].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        axes[1, i].imshow(recon_np)
        axes[1, i].set_title('Reconstruction', fontsize=10)
        axes[1, i].axis('off')
        
        # Error map
        if error_maps is not None and n_rows >= 3:
            err = error_maps[i]
            if isinstance(err, torch.Tensor):
                err = err.cpu().numpy()
            if err.ndim == 3:
                err = err.mean(axis=0)
            im = axes[2, i].imshow(err, cmap='hot', vmin=0)
            axes[2, i].set_title('Error Map', fontsize=10)
            axes[2, i].axis('off')
        
        # Ground truth mask
        if masks is not None and n_rows >= 4:
            mask = masks[i]
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            if mask.ndim == 3:
                mask = mask[0]
            axes[3, i].imshow(mask, cmap='gray')
            axes[3, i].set_title('Ground Truth', fontsize=10)
            axes[3, i].axis('off')
    
    # Row labels
    row_labels = ['Original', 'Reconstruction']
    if error_maps is not None:
        row_labels.append('Error Map')
    if masks is not None:
        row_labels.append('Ground Truth')
    
    for idx, label in enumerate(row_labels):
        axes[idx, 0].set_ylabel(label, fontsize=12, rotation=90, labelpad=20)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_anomaly_heatmap_overlay(
    image: torch.Tensor,
    error_map: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    threshold: Optional[float] = None,
    title: str = "Anomaly Detection",
    save_path: Optional[Path] = None,
):
    """
    Plot image with anomaly heatmap overlay.
    
    Args:
        image: Original image
        error_map: Error/anomaly map
        mask: Optional ground truth mask
        threshold: Optional threshold for binary prediction
        title: Plot title
        save_path: Path to save figure
    """
    n_cols = 4 if mask is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    
    img_np = denormalize_image(image)
    err_np = error_map.cpu().numpy() if isinstance(error_map, torch.Tensor) else error_map
    if err_np.ndim == 3:
        err_np = err_np.mean(axis=0)
    
    # Original
    axes[0].imshow(img_np)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(err_np, cmap='hot')
    axes[1].set_title('Anomaly Heatmap')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(img_np)
    axes[2].imshow(err_np, cmap='hot', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    # Ground truth
    if mask is not None:
        mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        if mask_np.ndim == 3:
            mask_np = mask_np[0]
        axes[3].imshow(mask_np, cmap='gray')
        axes[3].set_title('Ground Truth')
        axes[3].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "ROC Curves Comparison",
    save_path: Optional[Path] = None,
):
    """
    Plot multiple ROC curves for model comparison.
    
    Args:
        results: Dict of {model_name: (labels, scores)}
        title: Plot title
        save_path: Path to save figure
    """
    from sklearn.metrics import roc_auc_score
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    for (name, (labels, scores)), color in zip(results.items(), colors):
        auc = roc_auc_score(labels, scores)
        fpr, tpr, _ = roc_curve(labels, scores)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC = {auc:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_precision_recall_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "Precision-Recall Curves",
    save_path: Optional[Path] = None,
):
    """Plot Precision-Recall curves for multiple models."""
    from sklearn.metrics import average_precision_score
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    for (name, (labels, scores)), color in zip(results.items(), colors):
        ap = average_precision_score(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        ax.plot(recall, precision, color=color, linewidth=2, label=f'{name} (AP = {ap:.4f})')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training Curves",
    save_path: Optional[Path] = None,
):
    """
    Plot training loss curves.
    
    Args:
        history: Dict with 'train_loss', optionally 'val_loss', 'recon_loss', 'kl_loss'
        title: Plot title
        save_path: Path to save figure
    """
    valid_keys = [k for k, v in history.items() if 'loss' in k.lower() and len(v) > 0]
    n_plots = len(valid_keys)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]
    
    epochs = history.get('epochs', list(range(1, len(history['train_loss']) + 1)))
    
    plot_idx = 0
    for key in valid_keys:
        values = history[key]
        ax = axes[plot_idx]
        ax.plot(epochs, values, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(key.replace('_', ' ').title())
        ax.set_title(key.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
):
    """Plot confusion matrix with labels."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_score_distribution(
    normal_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    threshold: Optional[float] = None,
    title: str = "Anomaly Score Distribution",
    save_path: Optional[Path] = None,
):
    """
    Plot distribution of anomaly scores for normal vs anomalous samples.
    
    Args:
        normal_scores: Scores for normal samples
        anomaly_scores: Scores for anomalous samples
        threshold: Optional decision threshold
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='green', density=True)
    ax.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
    
    if threshold is not None:
        ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.4f}')
    
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_category_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = 'auc',
    title: str = "Per-Category Performance",
    save_path: Optional[Path] = None,
):
    """
    Plot bar chart comparing model performance across categories.
    
    Args:
        results: Dict of {model_name: {category: score}}
        metric: Metric name for label
        title: Plot title
        save_path: Path to save figure
    """
    models = list(results.keys())
    categories = list(results[models[0]].keys())
    
    x = np.arange(len(categories))
    width = 0.8 / len(models)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    for i, (model, color) in enumerate(zip(models, colors)):
        scores = [results[model][cat] for cat in categories]
        offset = (i - len(models)/2 + 0.5) * width
        ax.bar(x + offset, scores, width, label=model, color=color)
    
    ax.set_xlabel('Category')
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_latent_space_2d(
    latent_vectors: np.ndarray,
    labels: np.ndarray,
    method: str = 'pca',
    title: str = "Latent Space Visualization",
    save_path: Optional[Path] = None,
):
    """
    Visualize latent space using dimensionality reduction.
    
    Args:
        latent_vectors: Latent representations (N, dim)
        labels: Labels for coloring
        method: 'pca' or 'tsne'
        title: Plot title
        save_path: Path to save figure
    """
    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    
    coords_2d = reducer.fit_transform(latent_vectors)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=labels, 
                         cmap='coolwarm', alpha=0.7, s=30)
    
    plt.colorbar(scatter, ax=ax, label='Anomaly (1) / Normal (0)')
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def create_summary_figure(
    model_name: str,
    category: str,
    samples: List[Tuple],  # (original, recon, error, mask, label)
    roc_data: Tuple[np.ndarray, np.ndarray],
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
):
    """
    Create a comprehensive summary figure for a trained model.
    """
    from sklearn.metrics import roc_auc_score
    
    fig = plt.figure(figsize=(16, 12))
    
    n_samples = min(4, len(samples))
    
    # Training curve (top left)
    ax1 = fig.add_subplot(3, 3, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # ROC curve (top middle)
    ax2 = fig.add_subplot(3, 3, 2)
    labels, scores = roc_data
    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    ax2.plot(fpr, tpr, 'b-', linewidth=2)
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel('FPR')
    ax2.set_ylabel('TPR')
    ax2.set_title(f'ROC Curve (AUC = {auc:.4f})')
    ax2.grid(True, alpha=0.3)
    
    # Score distribution (top right)
    ax3 = fig.add_subplot(3, 3, 3)
    normal_mask = np.array(labels) == 0
    ax3.hist(np.array(scores)[normal_mask], bins=30, alpha=0.7, label='Normal', color='green', density=True)
    ax3.hist(np.array(scores)[~normal_mask], bins=30, alpha=0.7, label='Anomaly', color='red', density=True)
    ax3.set_xlabel('Score')
    ax3.set_ylabel('Density')
    ax3.set_title('Score Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Sample reconstructions (bottom rows)
    for i in range(min(n_samples, 4)):
        orig, recon, error, mask, label = samples[i]
        col = i
        
        # Original
        ax = fig.add_subplot(3, 4, 5 + col)
        ax.imshow(denormalize_image(orig))
        ax.set_title('Defect' if label else 'Normal', fontsize=9)
        ax.axis('off')
        
        # Error map
        ax = fig.add_subplot(3, 4, 9 + col)
        err = error.cpu().numpy() if isinstance(error, torch.Tensor) else error
        if err.ndim == 3:
            err = err.mean(axis=0)
        ax.imshow(err, cmap='hot')
        ax.set_title('Error', fontsize=9)
        ax.axis('off')
    
    plt.suptitle(f'{model_name} - {category.title()}', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
