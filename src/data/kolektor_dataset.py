"""
KolektorSDD2 Dataset Loader.

KolektorSDD2 is a real-world industrial dataset for surface defect detection
on electrical commutators. Used for generalization testing.

Dataset Structure:
    kolektor_sdd2/
    ├── train/
    │   ├── img/          # Training images (*.png)
    │   └── ann/          # Annotation masks (*.png)
    └── test/
        ├── img/          # Test images
        └── ann/          # Test annotations
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Literal

from PIL import Image
import torch
from torch.utils.data import Dataset

from src.config import KOLEKTOR_DIR
from src.data.transforms import get_transforms, get_mask_transforms


class KolektorDataset(Dataset):
    """
    PyTorch Dataset for KolektorSDD2.
    
    This dataset contains images of electrical commutators with surface defects.
    Each image has a corresponding binary mask indicating defect regions.
    
    Args:
        split: 'train' or 'test'
        transform: Optional transform to apply to images
        mask_transform: Optional transform for ground truth masks
        return_mask: If True, returns mask along with image
        
    Attributes:
        image_paths: List of paths to images
        mask_paths: List of paths to corresponding masks
        labels: List of labels (0=normal, 1=defective)
    """
    
    def __init__(
        self,
        split: Literal['train', 'test'] = 'train',
        transform=None,
        mask_transform=None,
        return_mask: bool = True,
    ):
        self.split = split
        self.transform = transform or get_transforms(train=(split == 'train'))
        self.mask_transform = mask_transform or get_mask_transforms()
        self.return_mask = return_mask
        
        self.base_dir = Path(KOLEKTOR_DIR) / split
        self.image_dir = self.base_dir / 'img'
        self.mask_dir = self.base_dir / 'ann'
        
        self.image_paths: List[Path] = []
        self.mask_paths: List[Path] = []
        self.labels: List[int] = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load image and mask paths from directory structure."""
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        # Get all images
        for img_name in sorted(os.listdir(self.image_dir)):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            img_path = self.image_dir / img_name
            mask_path = self.mask_dir / img_name
            
            self.image_paths.append(img_path)
            self.mask_paths.append(mask_path)
            
            # Determine label by checking if mask contains defects
            # For efficiency, we'll check this during __getitem__
            # Initially mark all as potentially defective
            self.labels.append(-1)  # -1 indicates not yet determined
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a sample from the dataset.
        
        Returns:
            If return_mask is False:
                (image, label)
            If return_mask is True:
                (image, mask, label)
        """
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Load mask
        mask_path = self.mask_paths[idx]
        if mask_path.exists():
            mask = Image.open(mask_path).convert('L')
            if self.mask_transform:
                mask = self.mask_transform(mask)
            # Determine label from mask
            label = 1 if mask.max() > 0 else 0
        else:
            mask = torch.zeros(1, image.shape[1], image.shape[2])
            label = 0
        
        if not self.return_mask:
            return image, label
        
        return image, mask, label
    
    def get_normal_indices(self) -> List[int]:
        """
        Get indices of normal (defect-free) samples.
        Useful for unsupervised training on normal samples only.
        """
        normal_indices = []
        for idx in range(len(self)):
            mask_path = self.mask_paths[idx]
            if mask_path.exists():
                mask = Image.open(mask_path).convert('L')
                if max(mask.getdata()) == 0:
                    normal_indices.append(idx)
            else:
                normal_indices.append(idx)
        return normal_indices


def create_kolektor_dataloaders(
    batch_size: int = 16,
    num_workers: int = 0,
    return_mask: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and test dataloaders for KolektorSDD2.
    
    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of data loading workers
        return_mask: If True, loaders return masks
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_dataset = KolektorDataset(
        split='train',
        return_mask=return_mask,
    )
    
    test_dataset = KolektorDataset(
        split='test',
        return_mask=return_mask,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    return train_loader, test_loader
