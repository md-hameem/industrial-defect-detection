"""
Image transforms for data preprocessing.

All transforms are designed for 256x256 input images
with ImageNet normalization for transfer learning compatibility.
"""

from torchvision import transforms
from src.config import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD


def get_transforms(train: bool = True, normalize: bool = True, augment_level: str = 'standard'):
    """
    Get image transforms for training or evaluation.
    
    Args:
        train: If True, includes data augmentation transforms
        normalize: If True, applies ImageNet normalization
        augment_level: 'none', 'standard', or 'strong' augmentation level
        
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    transform_list = []
    
    # Resize to target size
    transform_list.append(transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)))
    
    # Data augmentation for training
    if train and augment_level != 'none':
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ])
        
        if augment_level == 'strong':
            transform_list.extend([
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.1,
                    hue=0.02,
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                ),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            ])
    
    # Convert to tensor
    transform_list.append(transforms.ToTensor())
    
    # Normalize using ImageNet statistics
    if normalize:
        transform_list.append(
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        )
    
    return transforms.Compose(transform_list)


def get_mask_transforms():
    """
    Get transforms for ground truth masks.
    
    Masks are resized and converted to tensor without normalization.
    Uses nearest neighbor interpolation to preserve binary values.
    
    Returns:
        torchvision.transforms.Compose: Composed transforms for masks
    """
    return transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=transforms.InterpolationMode.NEAREST
        ),
        transforms.ToTensor(),
    ])


def denormalize(tensor):
    """
    Denormalize a tensor for visualization.
    
    Args:
        tensor: Normalized image tensor (C, H, W)
        
    Returns:
        Denormalized tensor suitable for display
    """
    import torch
    
    mean = torch.tensor(NORMALIZE_MEAN).view(3, 1, 1)
    std = torch.tensor(NORMALIZE_STD).view(3, 1, 1)
    
    return tensor * std + mean
