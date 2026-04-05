"""
Unit tests for evaluation metrics module.

Tests metric correctness with known inputs/outputs.
"""

import pytest
import numpy as np
from src.evaluation.metrics import (
    compute_roc_auc,
    compute_average_precision,
    compute_classification_metrics,
    compute_iou,
    compute_dice,
    compute_pixel_auc,
    find_optimal_threshold,
    compute_all_metrics,
)


class TestImageLevelMetrics:
    """Tests for image-level evaluation metrics."""
    
    def test_perfect_roc_auc(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        auc = compute_roc_auc(labels, scores)
        assert auc == 1.0
    
    def test_random_roc_auc(self):
        labels = np.array([0, 1, 0, 1])
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        auc = compute_roc_auc(labels, scores)
        assert auc == 0.5
    
    def test_single_class_auc_returns_default(self):
        labels = np.array([0, 0, 0])
        scores = np.array([0.1, 0.2, 0.3])
        auc = compute_roc_auc(labels, scores)
        assert auc == 0.5  # Cannot compute with single class
    
    def test_perfect_ap(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.9, 0.8])
        ap = compute_average_precision(labels, scores)
        assert ap == 1.0
    
    def test_classification_metrics_perfect(self):
        labels = np.array([0, 0, 1, 1])
        preds = np.array([0, 0, 1, 1])
        metrics = compute_classification_metrics(labels, preds)
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
        assert metrics['accuracy'] == 1.0
    
    def test_classification_metrics_empty(self):
        labels = np.array([])
        preds = np.array([])
        metrics = compute_classification_metrics(labels, preds)
        assert metrics['precision'] == 0.0


class TestPixelLevelMetrics:
    """Tests for pixel-level evaluation metrics."""
    
    def test_perfect_iou(self):
        pred = np.ones((64, 64))
        gt = np.ones((64, 64))
        iou = compute_iou(pred, gt, threshold=0.5)
        assert iou == 1.0
    
    def test_zero_iou(self):
        pred = np.zeros((64, 64))
        gt = np.ones((64, 64))
        iou = compute_iou(pred, gt, threshold=0.5)
        assert iou == 0.0
    
    def test_partial_iou(self):
        pred = np.zeros((64, 64))
        gt = np.zeros((64, 64))
        pred[:32, :32] = 1.0
        gt[:32, :32] = 1.0
        gt[32:, :32] = 1.0  # Extra GT region not predicted
        iou = compute_iou(pred, gt, threshold=0.5)
        assert 0.0 < iou < 1.0
    
    def test_perfect_dice(self):
        pred = np.ones((64, 64))
        gt = np.ones((64, 64))
        dice = compute_dice(pred, gt, threshold=0.5)
        assert dice == 1.0
    
    def test_both_empty_iou(self):
        pred = np.zeros((64, 64))
        gt = np.zeros((64, 64))
        iou = compute_iou(pred, gt, threshold=0.5)
        assert iou == 1.0  # Both empty = perfect agreement
    
    def test_pixel_auc(self):
        pred_masks = [np.random.rand(64, 64) for _ in range(10)]
        gt_masks = [(m > 0.5).astype(float) for m in pred_masks]
        auc = compute_pixel_auc(pred_masks, gt_masks)
        assert 0.0 <= auc <= 1.0


class TestThresholdOptimization:
    """Tests for threshold finding."""
    
    def test_finds_good_threshold(self):
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8, 0.9])
        threshold, metrics = find_optimal_threshold(labels, scores, method='f1')
        assert 0.3 <= threshold <= 0.7, f"Threshold {threshold} seems wrong"
        assert metrics['f1'] > 0.5
    
    def test_youden_method(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        threshold, metrics = find_optimal_threshold(labels, scores, method='youden')
        assert 0.3 <= threshold <= 0.7


class TestAllMetrics:
    """Tests for compute_all_metrics integration."""
    
    def test_image_only_metrics(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.2, 0.3, 0.7, 0.8])
        metrics = compute_all_metrics(labels, scores)
        assert 'image_auc' in metrics
        assert 'image_ap' in metrics
    
    def test_with_pixel_metrics(self):
        labels = np.array([0, 1])
        scores = np.array([0.2, 0.8])
        pred_masks = [np.zeros((64, 64)), np.ones((64, 64))]
        gt_masks = [np.zeros((64, 64)), np.ones((64, 64))]
        metrics = compute_all_metrics(labels, scores, pred_masks, gt_masks)
        assert 'pixel_auc' in metrics
        assert 'mean_iou' in metrics
        assert 'mean_dice' in metrics
