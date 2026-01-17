"""
KolektorSDD2 Dataset Loader.

KolektorSDD2 is a real-world industrial dataset for surface defect detection
on electrical commutators. Used for generalization testing.

Dataset Structure:
    kolektor_sdd2/
    ├── train/
    │   ├── img/          # Training images (*.png)
    │   └── ann/          # Annotation JSON files (*.png.json)
    └── test/
        ├── img/          # Test images
        └── ann/          # Test annotations (JSON format)
"""

import os
import json
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, List, Literal

from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset

from src.config import KOLEKTOR_DIR
from src.data.transforms import get_transforms, get_mask_transforms


class KolektorDataset(Dataset):
    """
    PyTorch Dataset for KolektorSDD2.
    
    This dataset contains images of electrical commutators with surface defects.
    Annotations are JSON files containing object polygons or bitmaps.
    
    Args:
        split: 'train' or 'test'
        transform: Optional transform to apply to images
        mask_transform: Optional transform for ground truth masks
        return_mask: If True, returns mask along with image
        
    Attributes:
        image_paths: List of paths to images
        ann_paths: List of paths to corresponding annotation JSONs
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
        self.ann_dir = self.base_dir / 'ann'
        
        self.image_paths: List[Path] = []
        self.ann_paths: List[Path] = []
        self.labels: List[int] = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load image and annotation paths from directory structure."""
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        # Get all images
        for img_name in sorted(os.listdir(self.image_dir)):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            img_path = self.image_dir / img_name
            # Annotation is named like "20000.png.json"
            ann_path = self.ann_dir / f"{img_name}.json"
            
            self.image_paths.append(img_path)
            self.ann_paths.append(ann_path)
            
            # Determine label from annotation
            if ann_path.exists():
                with open(ann_path, 'r') as f:
                    ann = json.load(f)
                label = 1 if len(ann.get('objects', [])) > 0 else 0
            else:
                label = 0
            
            self.labels.append(label)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def _decode_bitmap(self, bitmap_data: str, origin: List[int], img_size: Tuple[int, int]) -> np.ndarray:
        """Decode base64 bitmap from Supervisely format."""
        try:
            # Decode base64
            bitmap_bytes = base64.b64decode(bitmap_data)
            # Load as image
            bitmap_img = Image.open(BytesIO(bitmap_bytes))
            bitmap_arr = np.array(bitmap_img)
            
            # Create full mask
            h, w = img_size
            full_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Place bitmap at origin
            y0, x0 = origin
            bh, bw = bitmap_arr.shape[:2]
            y1 = min(y0 + bh, h)
            x1 = min(x0 + bw, w)
            
            if len(bitmap_arr.shape) > 2:
                bitmap_arr = bitmap_arr[:, :, 0]  # Take first channel
            
            full_mask[y0:y1, x0:x1] = bitmap_arr[:y1-y0, :x1-x0] > 0
            full_mask = full_mask * 255
            
            return full_mask
        except:
            return np.zeros(img_size, dtype=np.uint8)
    
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
        orig_size = (image.height, image.width)
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        if not self.return_mask:
            return image, label
        
        # Load annotation and create mask
        ann_path = self.ann_paths[idx]
        mask = np.zeros(orig_size, dtype=np.uint8)
        
        if ann_path.exists() and label == 1:
            with open(ann_path, 'r') as f:
                ann = json.load(f)
            
            for obj in ann.get('objects', []):
                if 'bitmap' in obj:
                    bitmap_data = obj['bitmap'].get('data', '')
                    origin = obj['bitmap'].get('origin', [0, 0])
                    obj_mask = self._decode_bitmap(bitmap_data, origin, orig_size)
                    mask = np.maximum(mask, obj_mask)
        
        # Convert to PIL for transform
        mask_pil = Image.fromarray(mask)
        if self.mask_transform:
            mask = self.mask_transform(mask_pil)
        else:
            mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        
        return image, mask, label


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
