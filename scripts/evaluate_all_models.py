"""
Generate comprehensive evaluation metrics report for all trained models.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DEVICE, MVTEC_CATEGORIES, MODELS_DIR, OUTPUTS_DIR
from src.data import create_mvtec_dataloaders
from src.models import create_cae, create_vae, create_denoising_ae
from src.evaluation import (
    compute_roc_auc,
    compute_average_precision,
    find_optimal_threshold,
    compute_iou,
    compute_dice,
    compute_pixel_auc,
    print_metrics,
)


def load_model(model_type: str, category: str, device: str = DEVICE):
    """Load a trained model."""
    model_path = MODELS_DIR / f"{model_type.lower()}_{category}_final.pth"
    
    if not model_path.exists():
        return None
    
    if model_type.upper() == "CAE":
        model = create_cae()
    elif model_type.upper() == "VAE":
        model = create_vae()
    elif model_type.upper() == "DAE":
        model = create_denoising_ae()
    else:
        return None
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    
    return model


def evaluate_category(model, dataloader, device: str = DEVICE, has_masks: bool = False):
    """Evaluate model on a single category."""
    model.eval()
    
    all_labels = []
    all_scores = []
    all_pred_masks = []
    all_gt_masks = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                images, masks, labels = batch
                has_masks = True
            elif len(batch) == 2:
                images, labels = batch
                masks = None
            else:
                continue
            
            images = images.to(device)
            
            # Get anomaly scores
            scores = model.get_anomaly_score(images)
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.numpy())
            
            # Get pixel-level predictions if masks available
            if has_masks and masks is not None:
                anomaly_maps = model.get_anomaly_map(images)
                for i in range(len(images)):
                    pred_map = anomaly_maps[i, 0].cpu().numpy()
                    # Normalize to 0-1
                    pred_map = (pred_map - pred_map.min()) / (pred_map.max() - pred_map.min() + 1e-8)
                    all_pred_masks.append(pred_map)
                    all_gt_masks.append(masks[i].numpy())
    
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # Compute metrics
    metrics = {}
    
    # Image-level
    if len(np.unique(all_labels)) > 1:
        metrics['image_auc'] = compute_roc_auc(all_labels, all_scores)
        metrics['image_ap'] = compute_average_precision(all_labels, all_scores)
        
        # Find optimal threshold
        threshold, threshold_metrics = find_optimal_threshold(all_labels, all_scores)
        metrics['optimal_threshold'] = threshold
        metrics['precision'] = threshold_metrics['precision']
        metrics['recall'] = threshold_metrics['recall']
        metrics['f1'] = threshold_metrics['f1']
    else:
        metrics['image_auc'] = 0.5
        metrics['image_ap'] = 0.0
        metrics['optimal_threshold'] = 0.5
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
        metrics['f1'] = 0.0
    
    # Pixel-level (if masks available)
    if all_pred_masks and all_gt_masks:
        try:
            metrics['pixel_auc'] = compute_pixel_auc(all_pred_masks, all_gt_masks)
        except:
            metrics['pixel_auc'] = 0.5
        
        # Mean IoU and Dice
        ious = []
        dices = []
        for pred, gt in zip(all_pred_masks, all_gt_masks):
            if gt.max() > 0:  # Only compute for images with defects
                ious.append(compute_iou(pred, gt, threshold=0.5))
                dices.append(compute_dice(pred, gt, threshold=0.5))
        
        metrics['mean_iou'] = np.mean(ious) if ious else 0.0
        metrics['mean_dice'] = np.mean(dices) if dices else 0.0
    
    return metrics


def main():
    print("="*60)
    print("  COMPREHENSIVE MODEL EVALUATION REPORT")
    print("="*60)
    
    results = []
    
    model_types = ["CAE", "VAE", "DAE"]
    
    for model_type in model_types:
        print(f"\n📊 Evaluating {model_type}...")
        
        for category in tqdm(MVTEC_CATEGORIES, desc=f"{model_type}"):
            model = load_model(model_type, category)
            if model is None:
                continue
            
            try:
                _, test_loader = create_mvtec_dataloaders(
                    category, batch_size=8, return_mask=True
                )
                
                metrics = evaluate_category(model, test_loader, DEVICE, has_masks=True)
                metrics['model'] = model_type
                metrics['category'] = category
                results.append(metrics)
                
            except Exception as e:
                print(f"  Error evaluating {model_type}/{category}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['model', 'category', 'image_auc', 'image_ap', 'precision', 'recall', 'f1']
    if 'pixel_auc' in df.columns:
        cols.extend(['pixel_auc', 'mean_iou', 'mean_dice'])
    cols.append('optimal_threshold')
    df = df[cols]
    
    # Save to CSV
    output_path = OUTPUTS_DIR / "comprehensive_metrics_report.csv"
    df.to_csv(output_path, index=False)
    print(f"\n📁 Saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("  SUMMARY BY MODEL")
    print("="*60)
    
    for model_type in model_types:
        model_df = df[df['model'] == model_type]
        if len(model_df) == 0:
            continue
        
        print(f"\n🔹 {model_type}:")
        print(f"   Mean Image AUC:  {model_df['image_auc'].mean():.4f}")
        print(f"   Mean Image AP:   {model_df['image_ap'].mean():.4f}")
        print(f"   Mean Precision:  {model_df['precision'].mean():.4f}")
        print(f"   Mean Recall:     {model_df['recall'].mean():.4f}")
        print(f"   Mean F1:         {model_df['f1'].mean():.4f}")
        if 'pixel_auc' in model_df.columns:
            print(f"   Mean Pixel AUC:  {model_df['pixel_auc'].mean():.4f}")
            print(f"   Mean IoU:        {model_df['mean_iou'].mean():.4f}")
            print(f"   Mean Dice:       {model_df['mean_dice'].mean():.4f}")
    
    # Best categories per model
    print("\n" + "="*60)
    print("  BEST CATEGORIES PER MODEL (by AUC)")
    print("="*60)
    
    for model_type in model_types:
        model_df = df[df['model'] == model_type]
        if len(model_df) == 0:
            continue
        
        best = model_df.loc[model_df['image_auc'].idxmax()]
        worst = model_df.loc[model_df['image_auc'].idxmin()]
        
        print(f"\n🔹 {model_type}:")
        print(f"   Best:  {best['category']} (AUC: {best['image_auc']:.4f})")
        print(f"   Worst: {worst['category']} (AUC: {worst['image_auc']:.4f})")
    
    print("\n" + "="*60)
    print("  EVALUATION COMPLETE!")
    print("="*60)
    
    return df


if __name__ == "__main__":
    main()
