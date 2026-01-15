"""
Configuration settings for Industrial Defect Detection project.

This module contains all configurable parameters for:
- Dataset paths
- Image preprocessing
- Training hyperparameters
- Model architecture settings
"""

import os
from pathlib import Path

# =============================================================================
# Project Paths
# =============================================================================

# Base project directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Dataset paths
DATASETS_DIR = PROJECT_ROOT / "datasets"
MVTEC_DIR = DATASETS_DIR / "mvtec_ad"
KOLEKTOR_DIR = DATASETS_DIR / "kolektor_sdd2"
NEU_DIR = DATASETS_DIR / "neu_surface_defect"

# Output paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
LOGS_DIR = OUTPUTS_DIR / "logs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# =============================================================================
# Image Settings
# =============================================================================

# Input image size (constrained by CPU training)
IMAGE_SIZE = 256

# Normalization parameters (ImageNet statistics)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# =============================================================================
# Dataset Categories
# =============================================================================

# MVTec AD categories
MVTEC_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

# MVTec texture vs object categories
MVTEC_TEXTURES = ['carpet', 'grid', 'leather', 'tile', 'wood']
MVTEC_OBJECTS = [c for c in MVTEC_CATEGORIES if c not in MVTEC_TEXTURES]

# NEU Surface Defect categories
NEU_CATEGORIES = [
    'crazing', 'inclusion', 'patches', 
    'pitted_surface', 'rolled-in_scale', 'scratches'
]

# =============================================================================
# Training Hyperparameters
# =============================================================================

# Device settings (CPU only due to AMD GPU limitation)
DEVICE = "cpu"

# Training parameters
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# Data loading
NUM_WORKERS = 0  # Set to 0 for Windows compatibility
PIN_MEMORY = False  # Disabled for CPU training

# Early stopping
EARLY_STOPPING_PATIENCE = 10

# Checkpointing
SAVE_EVERY_N_EPOCHS = 10

# =============================================================================
# Model Architecture Settings
# =============================================================================

# Autoencoder settings
LATENT_DIM = 128  # Latent space dimension for VAE
AE_CHANNELS = [32, 64, 128, 256]  # Channel progression in encoder/decoder

# Denoising Autoencoder
NOISE_FACTOR = 0.3  # Amount of noise to add during training

# =============================================================================
# Evaluation Settings
# =============================================================================

# Threshold for binary defect detection
ANOMALY_THRESHOLD = 0.5

# Number of samples for visualization
NUM_VIS_SAMPLES = 10

# =============================================================================
# Utility Functions
# =============================================================================

def ensure_dirs():
    """Create all necessary output directories."""
    for dir_path in [MODELS_DIR, LOGS_DIR, FIGURES_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

def get_category_dir(dataset: str, category: str) -> Path:
    """Get the directory path for a specific dataset category."""
    if dataset == "mvtec":
        return MVTEC_DIR / category
    elif dataset == "kolektor":
        return KOLEKTOR_DIR
    elif dataset == "neu":
        return NEU_DIR
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
