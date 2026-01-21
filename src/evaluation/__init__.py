"""Evaluation metrics and visualization"""
from .visualization import (
    set_style,
    denormalize_image,
    plot_reconstruction_grid,
    plot_anomaly_heatmap_overlay,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_training_curves,
    plot_confusion_matrix,
    plot_score_distribution,
    plot_category_comparison,
    plot_latent_space_2d,
    create_summary_figure,
)

from .metrics import (
    compute_roc_auc,
    compute_average_precision,
    find_optimal_threshold,
    compute_classification_metrics,
    compute_iou,
    compute_dice,
    compute_pixel_auc,
    compute_pro,
    compute_all_metrics,
    evaluate_model,
    print_metrics,
)

__all__ = [
    # Visualization
    'set_style',
    'denormalize_image',
    'plot_reconstruction_grid',
    'plot_anomaly_heatmap_overlay',
    'plot_roc_curves',
    'plot_precision_recall_curves',
    'plot_training_curves',
    'plot_confusion_matrix',
    'plot_score_distribution',
    'plot_category_comparison',
    'plot_latent_space_2d',
    'create_summary_figure',
    # Metrics
    'compute_roc_auc',
    'compute_average_precision',
    'find_optimal_threshold',
    'compute_classification_metrics',
    'compute_iou',
    'compute_dice',
    'compute_pixel_auc',
    'compute_pro',
    'compute_all_metrics',
    'evaluate_model',
    'print_metrics',
]
