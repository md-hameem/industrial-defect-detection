"""
PatchCore-inspired Feature-Based Anomaly Detection.

PatchCore (Roth et al., CVPR 2022) achieves state-of-the-art anomaly detection
by using features from a pretrained ImageNet model (no training required).

How it works:
    1. Extract patch-level features from a pretrained ResNet
    2. Build a "memory bank" of normal patch features (from training data)
    3. At test time, find the nearest neighbor distance for each patch
    4. High distance = anomaly

Key advantages over autoencoder-based methods:
    - No training required (only feature extraction on normal data)
    - Uses rich ImageNet features instead of learning from scratch
    - Achieves ~0.85+ AUC on MVTec AD (vs ~0.62 for CAE)
    - Pixel-precise anomaly localization

This is a simplified version suitable for CPU inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class PatchCoreModel(nn.Module):
    """
    PatchCore-inspired anomaly detector using pretrained features.
    
    Uses a frozen pretrained ResNet-18 as a feature extractor.
    Normal patch features are stored in a memory bank.
    Anomaly is detected by nearest-neighbor distance.
    
    Args:
        backbone: Pretrained backbone ('resnet18' or 'wide_resnet50')
        layers: Which ResNet layers to extract features from
        k: Number of nearest neighbors for scoring
        subsample_ratio: Ratio of features to keep in memory bank (for speed)
    """
    
    def __init__(
        self,
        backbone: str = 'resnet18',
        layers: List[str] = ['layer2', 'layer3'],
        k: int = 3,
        subsample_ratio: float = 0.1,
    ):
        super().__init__()
        
        self.k = k
        self.subsample_ratio = subsample_ratio
        self.layers = layers
        self.memory_bank = None
        self._feature_size = None
        
        # Load pretrained backbone
        import torchvision.models as models
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif backbone == 'wide_resnet50':
            self.backbone = models.wide_resnet50_2(weights=models.WideResNet50_2_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        
        # Hook storage for intermediate features
        self._features = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate features."""
        for layer_name in self.layers:
            layer = getattr(self.backbone, layer_name)
            layer.register_forward_hook(self._get_hook(layer_name))
    
    def _get_hook(self, name: str):
        def hook(module, input, output):
            self._features[name] = output
        return hook
    
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract and combine multi-scale features.
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Patch features (B, N_patches, feature_dim)
        """
        self._features.clear()
        
        with torch.no_grad():
            _ = self.backbone(x)
        
        # Get features from specified layers
        features = []
        target_size = None
        
        for layer_name in self.layers:
            feat = self._features[layer_name]
            
            if target_size is None:
                target_size = feat.shape[2:]  # Use first layer's spatial size
            
            # Resize to common spatial size
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            
            features.append(feat)
        
        # Concatenate along channel dimension
        combined = torch.cat(features, dim=1)  # (B, C_combined, H_feat, W_feat)
        
        # Reshape to patch features: (B, H*W, C)
        B, C, H, W = combined.shape
        patch_features = combined.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        self._feature_size = (H, W, C)
        
        return patch_features
    
    def fit(self, dataloader: torch.utils.data.DataLoader, device: str = 'cpu'):
        """
        Build memory bank from normal training data.
        
        Args:
            dataloader: DataLoader with normal images only
            device: Device for computation
        """
        self.backbone.to(device)
        all_features = []
        
        print("Building PatchCore memory bank...")
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images = batch[0].to(device)
            else:
                images = batch.to(device)
            
            features = self._extract_features(images)
            all_features.append(features.cpu())
        
        # Combine all features
        all_features = torch.cat(all_features, dim=0)  # (N_images, N_patches, C)
        all_features = all_features.reshape(-1, all_features.shape[-1])  # (N_total_patches, C)
        
        # Subsample memory bank for efficiency (coreset selection simplified)
        n_total = all_features.shape[0]
        n_keep = max(1, int(n_total * self.subsample_ratio))
        
        # Random subsampling (simplified version of greedy coreset)
        indices = torch.randperm(n_total)[:n_keep]
        self.memory_bank = all_features[indices]
        
        print(f"Memory bank: {self.memory_bank.shape[0]} patches "
              f"(subsampled from {n_total}, feature dim={self.memory_bank.shape[1]})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction (not applicable — returns input for API compat).
        For PatchCore, use get_anomaly_score() and get_anomaly_map() directly.
        """
        return x
    
    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute image-level anomaly scores.
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Anomaly scores (B,) — higher = more anomalous
        """
        with torch.no_grad():
            anomaly_maps = self._compute_patch_distances(x)
            # Image-level score = max patch distance
            scores = anomaly_maps.reshape(anomaly_maps.shape[0], -1).max(dim=1)[0]
        return scores
    
    def get_anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate pixel-wise anomaly map.
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Anomaly maps (B, 1, H, W) — upsampled to input resolution
        """
        with torch.no_grad():
            patch_distances = self._compute_patch_distances(x)
            
            # Reshape to spatial map
            B = x.shape[0]
            H, W, _ = self._feature_size
            anomaly_map = patch_distances.reshape(B, 1, H, W)
            
            # Upsample to input resolution
            anomaly_map = F.interpolate(
                anomaly_map, size=x.shape[2:], mode='bilinear', align_corners=False
            )
        
        return anomaly_map
    
    def _compute_patch_distances(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute distance of each test patch to nearest memory bank patches.
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Patch distances (B, N_patches)
        """
        if self.memory_bank is None:
            raise RuntimeError("Memory bank not built. Call fit() first.")
        
        features = self._extract_features(x)  # (B, N_patches, C)
        memory = self.memory_bank.to(features.device)  # (M, C)
        
        B, N, C = features.shape
        M = memory.shape[0]
        
        # Compute distances in chunks to avoid OOM
        distances = []
        chunk_size = 1000  # Process 1000 memory bank entries at a time
        
        for b in range(B):
            test_feats = features[b]  # (N, C)
            min_dists = torch.full((N,), float('inf'), device=features.device)
            
            for start in range(0, M, chunk_size):
                end = min(start + chunk_size, M)
                mem_chunk = memory[start:end]  # (chunk, C)
                
                # Pairwise L2 distances: (N, chunk)
                dists = torch.cdist(test_feats.unsqueeze(0), mem_chunk.unsqueeze(0))[0]
                
                # Top-k minimum distances
                k = min(self.k, dists.shape[1])
                top_k_dists = dists.topk(k, dim=1, largest=False)[0]
                batch_min = top_k_dists.mean(dim=1)  # Average of k-nearest
                
                min_dists = torch.min(min_dists, batch_min)
            
            distances.append(min_dists)
        
        return torch.stack(distances, dim=0)  # (B, N_patches)
    
    def save_memory_bank(self, path: str):
        """Save memory bank to file."""
        if self.memory_bank is None:
            raise RuntimeError("No memory bank to save")
        
        torch.save({
            'memory_bank': self.memory_bank,
            'feature_size': self._feature_size,
            'k': self.k,
            'subsample_ratio': self.subsample_ratio,
            'layers': self.layers,
        }, path)
        print(f"Saved memory bank to {path}")
    
    def load_memory_bank(self, path: str):
        """Load memory bank from file."""
        data = torch.load(path, map_location='cpu', weights_only=True)
        self.memory_bank = data['memory_bank']
        self._feature_size = data['feature_size']
        self.k = data['k']
        self.subsample_ratio = data['subsample_ratio']
        print(f"Loaded memory bank: {self.memory_bank.shape[0]} patches")


def create_patchcore(
    backbone: str = 'resnet18',
    k: int = 3,
    subsample_ratio: float = 0.1,
) -> PatchCoreModel:
    """
    Factory function to create PatchCore model.
    
    Args:
        backbone: Pretrained backbone name
        k: Number of nearest neighbors
        subsample_ratio: Memory bank subsampling ratio
        
    Returns:
        PatchCoreModel
    """
    return PatchCoreModel(
        backbone=backbone,
        k=k,
        subsample_ratio=subsample_ratio,
    )


if __name__ == "__main__":
    # Quick test
    model = create_patchcore()
    x = torch.randn(2, 3, 256, 256)
    
    # Simulate building memory bank from "normal" data
    from torch.utils.data import DataLoader, TensorDataset
    normal_data = TensorDataset(torch.randn(10, 3, 256, 256))
    loader = DataLoader(normal_data, batch_size=4)
    model.fit(loader)
    
    # Test inference
    score = model.get_anomaly_score(x)
    print(f"Anomaly scores: {score}")
    
    amap = model.get_anomaly_map(x)
    print(f"Anomaly map shape: {amap.shape}")
    
    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {param_count:,} (trainable: {trainable})")
