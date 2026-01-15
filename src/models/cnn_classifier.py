"""
Lightweight CNN Classifier for Supervised Baseline.

This classifier is used for the NEU Surface Defect dataset
as a supervised comparison baseline for the unsupervised methods.

Architecture:
    Input -> Conv blocks -> Global Average Pooling -> FC -> Softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ConvBlock(nn.Module):
    """Convolutional block with Conv, BatchNorm, ReLU, and optional pooling."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool: bool = True,
    ):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightweightCNN(nn.Module):
    """
    Lightweight CNN classifier for defect classification.
    
    Designed for CPU efficiency while maintaining good accuracy.
    Uses Global Average Pooling to reduce parameters.
    
    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels (3 for RGB)
        channels: List of channel sizes for conv layers
        dropout: Dropout probability before final layer
        
    Input shape: (B, 3, 256, 256)
    Output shape: (B, num_classes)
    """
    
    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 3,
        channels: List[int] = [32, 64, 128, 256],
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Convolutional feature extractor
        layers = []
        prev_channels = in_channels
        
        for ch in channels:
            layers.append(ConvBlock(prev_channels, ch, pool=True))
            prev_channels = ch
        
        self.features = nn.Sequential(*layers)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class logits."""
        features = self.features(x)
        pooled = self.gap(features)
        flattened = pooled.view(pooled.size(0), -1)
        logits = self.classifier(flattened)
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class indices."""
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head."""
        with torch.no_grad():
            features = self.features(x)
            pooled = self.gap(features)
            return pooled.view(pooled.size(0), -1)


def create_cnn_classifier(
    num_classes: int = 6,
    channels: List[int] = [32, 64, 128, 256],
    dropout: float = 0.5,
) -> LightweightCNN:
    """
    Factory function to create CNN classifier.
    
    Args:
        num_classes: Number of output classes
        channels: Channel sizes for conv layers
        dropout: Dropout probability
        
    Returns:
        LightweightCNN model
    """
    return LightweightCNN(
        num_classes=num_classes,
        in_channels=3,
        channels=channels,
        dropout=dropout,
    )


if __name__ == "__main__":
    # Quick test
    model = create_cnn_classifier(num_classes=6)
    x = torch.randn(2, 3, 256, 256)
    
    # Forward pass
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Predictions
    preds = model.predict(x)
    probs = model.predict_proba(x)
    print(f"Predictions: {preds}")
    print(f"Probabilities shape: {probs.shape}")
    
    # Features
    features = model.get_features(x)
    print(f"Feature shape: {features.shape}")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
