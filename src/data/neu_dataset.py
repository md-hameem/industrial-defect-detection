"""
NEU Surface Defect Database Loader.

NEU Surface Defect Database is a supervised classification dataset
containing steel surface defects in 6 categories. Used as supervised baseline.

Dataset Structure:
    neu_surface_defect/
    ├── train/
    │   ├── images/
    │   │   ├── crazing/
    │   │   ├── inclusion/
    │   │   ├── patches/
    │   │   ├── pitted_surface/
    │   │   ├── rolled-in_scale/
    │   │   └── scratches/
    │   └── annotations/
    │       └── ... (same structure)
    └── validation/
        ├── images/
        └── annotations/

Categories:
    - crazing
    - inclusion
    - patches
    - pitted_surface
    - rolled-in_scale
    - scratches
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Literal

from PIL import Image
import torch
from torch.utils.data import Dataset

from src.config import NEU_DIR, NEU_CATEGORIES
from src.data.transforms import get_transforms, get_mask_transforms


class NEUDataset(Dataset):
    """
    PyTorch Dataset for NEU Surface Defect Database.
    
    This is a supervised classification dataset with 6 defect categories.
    Each image belongs to one defect class.
    
    Args:
        split: 'train' or 'validation'
        transform: Optional transform to apply to images
        mask_transform: Optional transform for annotation masks
        return_mask: If True, returns mask along with image
        categories: Optional list of categories to include (default: all)
        
    Attributes:
        image_paths: List of paths to images
        mask_paths: List of paths to annotation masks
        labels: List of category indices
        category_names: List of category names corresponding to labels
    """
    
    def __init__(
        self,
        split: Literal['train', 'validation'] = 'train',
        transform=None,
        mask_transform=None,
        return_mask: bool = False,
        categories: Optional[List[str]] = None,
    ):
        self.split = split
        self.transform = transform or get_transforms(train=(split == 'train'))
        self.mask_transform = mask_transform or get_mask_transforms()
        self.return_mask = return_mask
        self.categories = categories or NEU_CATEGORIES
        
        self.base_dir = Path(NEU_DIR) / split
        self.image_base = self.base_dir / 'images'
        self.annotation_base = self.base_dir / 'annotations'
        
        self.image_paths: List[Path] = []
        self.mask_paths: List[Optional[Path]] = []
        self.labels: List[int] = []
        self.category_names: List[str] = []
        
        # Create category to index mapping
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.idx_to_category = {idx: cat for cat, idx in self.category_to_idx.items()}
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load image and annotation paths from directory structure."""
        if not self.image_base.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_base}")
        
        for category in self.categories:
            category_img_dir = self.image_base / category
            category_ann_dir = self.annotation_base / category if self.annotation_base.exists() else None
            
            if not category_img_dir.exists():
                print(f"Warning: Category directory not found: {category_img_dir}")
                continue
            
            category_idx = self.category_to_idx[category]
            
            for img_name in sorted(os.listdir(category_img_dir)):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue
                
                img_path = category_img_dir / img_name
                self.image_paths.append(img_path)
                self.labels.append(category_idx)
                self.category_names.append(category)
                
                # Try to find corresponding annotation
                if category_ann_dir and category_ann_dir.exists():
                    # Try different extensions for annotation
                    ann_name_base = img_path.stem
                    ann_path = None
                    for ext in ['.png', '.bmp', '.jpg']:
                        potential_path = category_ann_dir / f"{ann_name_base}{ext}"
                        if potential_path.exists():
                            ann_path = potential_path
                            break
                    self.mask_paths.append(ann_path)
                else:
                    self.mask_paths.append(None)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a sample from the dataset.
        
        Returns:
            If return_mask is False:
                (image, label) where label is category index
            If return_mask is True:
                (image, mask, label)
        """
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        if not self.return_mask:
            return image, label
        
        # Load mask if available
        mask_path = self.mask_paths[idx]
        if mask_path is not None and mask_path.exists():
            mask = Image.open(mask_path).convert('L')
            if self.mask_transform:
                mask = self.mask_transform(mask)
        else:
            mask = torch.zeros(1, image.shape[1], image.shape[2])
        
        return image, mask, label
    
    def get_category_name(self, label: int) -> str:
        """Get category name from label index."""
        return self.idx_to_category[label]
    
    def get_num_classes(self) -> int:
        """Get number of classes."""
        return len(self.categories)
    
    @staticmethod
    def get_categories() -> List[str]:
        """Get list of all NEU categories."""
        return NEU_CATEGORIES.copy()


def create_neu_dataloaders(
    batch_size: int = 16,
    num_workers: int = 0,
    return_mask: bool = False,
    categories: Optional[List[str]] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders for NEU Surface Defect Database.
    
    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of data loading workers
        return_mask: If True, loaders return masks
        categories: Optional list of categories to include
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = NEUDataset(
        split='train',
        return_mask=return_mask,
        categories=categories,
    )
    
    val_dataset = NEUDataset(
        split='validation',
        return_mask=return_mask,
        categories=categories,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    return train_loader, val_loader
