"""
MVTec Anomaly Detection Dataset Loader.

MVTec AD is the primary benchmark dataset containing 15 categories
of industrial objects and textures with pixel-level annotations.

Dataset Structure:
    mvtec_ad/
    ├── bottle/
    │   ├── train/
    │   │   └── good/          # Normal training images
    │   ├── test/
    │   │   ├── good/          # Normal test images
    │   │   ├── broken_large/  # Defect type 1
    │   │   └── broken_small/  # Defect type 2
    │   └── ground_truth/
    │       ├── broken_large/  # Pixel masks for defects
    │       └── broken_small/
    └── ...
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Literal

from PIL import Image
import torch
from torch.utils.data import Dataset

from src.config import MVTEC_DIR, MVTEC_CATEGORIES
from src.data.transforms import get_transforms, get_mask_transforms


class MVTecDataset(Dataset):
    """
    PyTorch Dataset for MVTec Anomaly Detection.
    
    For unsupervised training, only 'good' (normal) images are used.
    For evaluation, both normal and anomalous images are included.
    
    Args:
        category: MVTec category name (e.g., 'bottle', 'carpet')
        split: 'train' or 'test'
        transform: Optional transform to apply to images
        mask_transform: Optional transform for ground truth masks
        return_mask: If True, returns mask along with image (test only)
        
    Attributes:
        image_paths: List of paths to images
        mask_paths: List of paths to masks (None for normal images)
        labels: List of labels (0=normal, 1=anomaly)
    """
    
    def __init__(
        self,
        category: str,
        split: Literal['train', 'test'] = 'train',
        transform=None,
        mask_transform=None,
        return_mask: bool = False,
    ):
        if category not in MVTEC_CATEGORIES:
            raise ValueError(
                f"Unknown category: {category}. "
                f"Must be one of {MVTEC_CATEGORIES}"
            )
        
        self.category = category
        self.split = split
        self.transform = transform or get_transforms(train=(split == 'train'))
        self.mask_transform = mask_transform or get_mask_transforms()
        self.return_mask = return_mask and (split == 'test')
        
        self.category_dir = Path(MVTEC_DIR) / category
        self.image_paths: List[Path] = []
        self.mask_paths: List[Optional[Path]] = []
        self.labels: List[int] = []
        self.defect_types: List[str] = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load image and mask paths from directory structure."""
        split_dir = self.category_dir / self.split
        gt_dir = self.category_dir / 'ground_truth'
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        # Iterate through all defect type folders
        for defect_type in sorted(os.listdir(split_dir)):
            defect_dir = split_dir / defect_type
            
            if not defect_dir.is_dir():
                continue
            
            # Get all images in this defect folder
            for img_name in sorted(os.listdir(defect_dir)):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                img_path = defect_dir / img_name
                self.image_paths.append(img_path)
                self.defect_types.append(defect_type)
                
                # Normal images have no mask
                if defect_type == 'good':
                    self.labels.append(0)
                    self.mask_paths.append(None)
                else:
                    self.labels.append(1)
                    # Find corresponding mask
                    mask_name = img_name.replace('.png', '_mask.png')
                    mask_path = gt_dir / defect_type / mask_name
                    
                    if mask_path.exists():
                        self.mask_paths.append(mask_path)
                    else:
                        # Try without _mask suffix
                        mask_path = gt_dir / defect_type / img_name
                        self.mask_paths.append(mask_path if mask_path.exists() else None)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a sample from the dataset.
        
        Returns:
            If return_mask is False:
                (image, label)
            If return_mask is True:
                (image, mask, label) where mask is zeros for normal images
        """
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        if not self.return_mask:
            return image, label
        
        # Load or create mask
        mask_path = self.mask_paths[idx]
        if mask_path is not None and mask_path.exists():
            mask = Image.open(mask_path).convert('L')
            if self.mask_transform:
                mask = self.mask_transform(mask)
        else:
            # Create empty mask for normal images
            mask = torch.zeros(1, image.shape[1], image.shape[2])
        
        return image, mask, label
    
    def get_defect_type(self, idx: int) -> str:
        """Get the defect type for a sample."""
        return self.defect_types[idx]
    
    @staticmethod
    def get_categories() -> List[str]:
        """Get list of all MVTec categories."""
        return MVTEC_CATEGORIES.copy()


def create_mvtec_dataloaders(
    category: str,
    batch_size: int = 16,
    num_workers: int = 0,
    return_mask: bool = False,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and test dataloaders for an MVTec category.
    
    Args:
        category: MVTec category name
        batch_size: Batch size for dataloaders
        num_workers: Number of data loading workers
        return_mask: If True, test loader returns masks
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_dataset = MVTecDataset(
        category=category,
        split='train',
        return_mask=False,
    )
    
    test_dataset = MVTecDataset(
        category=category,
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
