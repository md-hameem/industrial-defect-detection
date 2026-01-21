"""
Evaluation Metrics for Anomaly Detection.

Includes:
- Image-level metrics: ROC-AUC, Precision, Recall, F1
- Pixel-level metrics: IoU, Dice, Pixel AUC
- Optimal threshold finding
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
from typing import Dict, List, Tuple, Optional
import torch


def compute_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute ROC-AUC score for image-level anomaly detection.
    
    Args:
        labels: Ground truth labels (0=normal, 1=anomaly)
        scores: Anomaly scores (higher = more anomalous)
        
    Returns:
        ROC-AUC score
    """
    if len(np.unique(labels)) < 2:
        return 0.5  # Cannot compute AUC with single class
    return roc_auc_score(labels, scores)


def compute_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute Average Precision (AP) score.
    
    Args:
        labels: Ground truth labels (0=normal, 1=anomaly)
        scores: Anomaly scores (higher = more anomalous)
        
    Returns:
        Average Precision score
    """
    if len(np.unique(labels)) < 2:
        return 0.0
    return average_precision_score(labels, scores)


def find_optimal_threshold(
    labels: np.ndarray, 
    scores: np.ndarray, 
    method: str = 'f1'
) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal threshold for classification.
    
    Args:
        labels: Ground truth labels
        scores: Anomaly scores
        method: Optimization method ('f1', 'youden', 'precision', 'recall')
        
    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold)
    """
    if method == 'youden':
        # Youden's J statistic: maximize (TPR - FPR)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[best_idx]
    else:
        # Grid search for best F1/Precision/Recall
        precision_arr, recall_arr, thresholds = precision_recall_curve(labels, scores)
        f1_scores = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-8)
        
        if method == 'f1':
            best_idx = np.argmax(f1_scores[:-1])  # Last element is undefined
        elif method == 'precision':
            # Find threshold with precision >= 0.9 and best recall
            valid = precision_arr[:-1] >= 0.9
            if valid.any():
                best_idx = np.argmax(recall_arr[:-1] * valid)
            else:
                best_idx = np.argmax(precision_arr[:-1])
        elif method == 'recall':
            # Find threshold with recall >= 0.9 and best precision
            valid = recall_arr[:-1] >= 0.9
            if valid.any():
                best_idx = np.argmax(precision_arr[:-1] * valid)
            else:
                best_idx = np.argmax(recall_arr[:-1])
        else:
            best_idx = np.argmax(f1_scores[:-1])
            
        optimal_threshold = thresholds[best_idx]
    
    # Compute metrics at optimal threshold
    predictions = (scores >= optimal_threshold).astype(int)
    metrics = compute_classification_metrics(labels, predictions)
    metrics['threshold'] = float(optimal_threshold)
    
    return optimal_threshold, metrics


def compute_classification_metrics(
    labels: np.ndarray, 
    predictions: np.ndarray
) -> Dict[str, float]:
    """
    Compute classification metrics from binary predictions.
    
    Args:
        labels: Ground truth labels (0=normal, 1=anomaly)
        predictions: Binary predictions (0=normal, 1=anomaly)
        
    Returns:
        Dictionary with precision, recall, f1, accuracy
    """
    # Handle edge cases
    if len(predictions) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0}
    
    accuracy = np.mean(labels == predictions)
    
    # Use zero_division=0 to handle cases with no positive predictions
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
    }


def compute_iou(
    pred_mask: np.ndarray, 
    gt_mask: np.ndarray, 
    threshold: float = 0.5
) -> float:
    """
    Compute Intersection over Union (IoU/Jaccard Index) for pixel-level.
    
    Args:
        pred_mask: Predicted anomaly mask (H, W) with values 0-1
        gt_mask: Ground truth mask (H, W) binary
        threshold: Threshold to binarize prediction
        
    Returns:
        IoU score
    """
    pred_binary = (pred_mask >= threshold).astype(bool)
    gt_binary = gt_mask.astype(bool)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float(intersection / union)


def compute_dice(
    pred_mask: np.ndarray, 
    gt_mask: np.ndarray, 
    threshold: float = 0.5
) -> float:
    """
    Compute Dice coefficient (F1 for pixel level).
    
    Args:
        pred_mask: Predicted anomaly mask (H, W) with values 0-1
        gt_mask: Ground truth mask (H, W) binary
        threshold: Threshold to binarize prediction
        
    Returns:
        Dice coefficient
    """
    pred_binary = (pred_mask >= threshold).astype(bool)
    gt_binary = gt_mask.astype(bool)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    total = pred_binary.sum() + gt_binary.sum()
    
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float(2 * intersection / total)


def compute_pixel_auc(
    pred_masks: List[np.ndarray], 
    gt_masks: List[np.ndarray]
) -> float:
    """
    Compute pixel-level ROC-AUC across all images.
    
    Args:
        pred_masks: List of predicted anomaly maps (H, W)
        gt_masks: List of ground truth masks (H, W)
        
    Returns:
        Pixel-level AUC
    """
    all_preds = np.concatenate([m.flatten() for m in pred_masks])
    all_gts = np.concatenate([m.flatten() for m in gt_masks])
    
    if len(np.unique(all_gts)) < 2:
        return 0.5
    
    return roc_auc_score(all_gts.astype(int), all_preds)


def compute_pro(
    pred_masks: List[np.ndarray],
    gt_masks: List[np.ndarray],
    num_thresholds: int = 200
) -> float:
    """
    Compute Per-Region Overlap (PRO) score.
    
    PRO is the average relative overlap of predicted abnormal regions
    with ground truth, computed across multiple thresholds.
    
    Args:
        pred_masks: List of predicted anomaly maps
        gt_masks: List of ground truth masks
        num_thresholds: Number of thresholds to evaluate
        
    Returns:
        PRO score (0-1, higher is better)
    """
    from scipy.ndimage import label
    
    thresholds = np.linspace(0, 1, num_thresholds)
    pro_scores = []
    
    for threshold in thresholds:
        overlaps = []
        for pred, gt in zip(pred_masks, gt_masks):
            pred_binary = pred >= threshold
            gt_binary = gt.astype(bool)
            
            # Find connected components in ground truth
            labeled_gt, num_regions = label(gt_binary)
            
            for region_id in range(1, num_regions + 1):
                region_mask = labeled_gt == region_id
                region_size = region_mask.sum()
                
                if region_size > 0:
                    overlap = np.logical_and(pred_binary, region_mask).sum()
                    overlaps.append(overlap / region_size)
        
        if overlaps:
            pro_scores.append(np.mean(overlaps))
    
    return float(np.mean(pro_scores)) if pro_scores else 0.0


def compute_all_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    pred_masks: Optional[List[np.ndarray]] = None,
    gt_masks: Optional[List[np.ndarray]] = None,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        labels: Image-level ground truth labels
        scores: Image-level anomaly scores
        pred_masks: Optional pixel-level predictions
        gt_masks: Optional pixel-level ground truth
        threshold: Optional threshold (auto-computed if None)
        
    Returns:
        Dictionary with all computed metrics
    """
    metrics = {}
    
    # Image-level metrics
    metrics['image_auc'] = compute_roc_auc(labels, scores)
    metrics['image_ap'] = compute_average_precision(labels, scores)
    
    # Find optimal threshold if not provided
    if threshold is None:
        threshold, threshold_metrics = find_optimal_threshold(labels, scores, method='f1')
        metrics.update({f'optimal_{k}': v for k, v in threshold_metrics.items()})
    else:
        predictions = (scores >= threshold).astype(int)
        threshold_metrics = compute_classification_metrics(labels, predictions)
        metrics.update(threshold_metrics)
    
    # Pixel-level metrics
    if pred_masks is not None and gt_masks is not None:
        metrics['pixel_auc'] = compute_pixel_auc(pred_masks, gt_masks)
        
        # Average IoU and Dice
        ious = [compute_iou(p, g, threshold=0.5) for p, g in zip(pred_masks, gt_masks)]
        dices = [compute_dice(p, g, threshold=0.5) for p, g in zip(pred_masks, gt_masks)]
        
        metrics['mean_iou'] = float(np.mean(ious))
        metrics['mean_dice'] = float(np.mean(dices))
        
        # PRO score (can be slow, only compute if needed)
        # metrics['pro'] = compute_pro(pred_masks, gt_masks)
    
    return metrics


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cpu',
    compute_pixel_metrics: bool = True
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: Trained model with get_anomaly_score and get_anomaly_map methods
        dataloader: DataLoader yielding (images, [masks], labels)
        device: Device to run inference on
        compute_pixel_metrics: Whether to compute pixel-level metrics
        
    Returns:
        Dictionary with all evaluation metrics
    """
    model.eval()
    
    all_labels = []
    all_scores = []
    all_pred_masks = []
    all_gt_masks = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Handle different batch formats
            if len(batch) == 3:
                images, masks, labels = batch
            elif len(batch) == 2:
                images, labels = batch
                masks = None
            else:
                continue
            
            images = images.to(device)
            
            # Get image-level scores
            scores = model.get_anomaly_score(images)
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.numpy())
            
            # Get pixel-level predictions if masks available
            if compute_pixel_metrics and masks is not None:
                anomaly_maps = model.get_anomaly_map(images)
                for i in range(len(images)):
                    all_pred_masks.append(anomaly_maps[i, 0].cpu().numpy())
                    all_gt_masks.append(masks[i].numpy())
    
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    return compute_all_metrics(
        all_labels, 
        all_scores,
        all_pred_masks if compute_pixel_metrics else None,
        all_gt_masks if compute_pixel_metrics else None
    )


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Results"):
    """Pretty print evaluation metrics."""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")
    
    # Image-level
    print("\n📊 Image-Level Metrics:")
    if 'image_auc' in metrics:
        print(f"   ROC-AUC:    {metrics['image_auc']:.4f}")
    if 'image_ap' in metrics:
        print(f"   AP:         {metrics['image_ap']:.4f}")
    
    # Classification at threshold
    if 'optimal_threshold' in metrics:
        print(f"\n🎯 At Optimal Threshold ({metrics['optimal_threshold']:.4f}):")
        print(f"   Precision:  {metrics.get('optimal_precision', 0):.4f}")
        print(f"   Recall:     {metrics.get('optimal_recall', 0):.4f}")
        print(f"   F1:         {metrics.get('optimal_f1', 0):.4f}")
    elif 'precision' in metrics:
        print(f"\n🎯 Classification Metrics:")
        print(f"   Precision:  {metrics['precision']:.4f}")
        print(f"   Recall:     {metrics['recall']:.4f}")
        print(f"   F1:         {metrics['f1']:.4f}")
    
    # Pixel-level
    if 'pixel_auc' in metrics:
        print("\n🔍 Pixel-Level Metrics:")
        print(f"   Pixel AUC:  {metrics['pixel_auc']:.4f}")
    if 'mean_iou' in metrics:
        print(f"   Mean IoU:   {metrics['mean_iou']:.4f}")
    if 'mean_dice' in metrics:
        print(f"   Mean Dice:  {metrics['mean_dice']:.4f}")
    
    print(f"{'='*50}\n")
