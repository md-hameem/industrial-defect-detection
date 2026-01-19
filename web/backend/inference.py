"""
Model Inference Module

Handles loading trained models and running inference on images.
"""

import sys
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add thesis project to path
THESIS_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(THESIS_ROOT))

from src.config import MODELS_DIR, DEVICE, MVTEC_CATEGORIES
from src.models import create_cae, create_vae, create_denoising_ae
from src.data.transforms import get_transforms, denormalize


class ModelInference:
    """
    Handles model loading and inference for web API.
    """
    
    def __init__(self):
        self.models: Dict[str, torch.nn.Module] = {}
        self.transform = get_transforms(train=False)
        self.models_dir = MODELS_DIR
        self.device = DEVICE
        
    def _get_model_path(self, model_type: str, category: str) -> Path:
        """Get path to saved model."""
        prefix_map = {
            "CAE": "cae",
            "VAE": "vae",
            "DAE": "dae",
        }
        prefix = prefix_map.get(model_type.upper(), "cae")
        return self.models_dir / f"{prefix}_{category}_final.pth"
    
    def _load_model(self, model_type: str, category: str) -> torch.nn.Module:
        """Load a trained model."""
        model_path = self._get_model_path(model_type, category)
        
        if not model_path.exists():
            raise ValueError(f"Model not found: {model_path}")
        
        # Create model
        if model_type.upper() == "CAE":
            model = create_cae()
        elif model_type.upper() == "VAE":
            model = create_vae()
        elif model_type.upper() == "DAE":
            model = create_denoising_ae()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.to(self.device)
        
        return model
    
    def _get_cached_model(self, model_type: str, category: str) -> torch.nn.Module:
        """Get model from cache or load it."""
        key = f"{model_type}_{category}"
        if key not in self.models:
            self.models[key] = self._load_model(model_type, category)
        return self.models[key]
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string."""
        # Ensure proper format
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def _create_heatmap(self, error_map: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Create heatmap overlay on original image."""
        fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
        ax.imshow(original)
        ax.imshow(error_map, cmap='jet', alpha=0.5)
        ax.axis('off')
        
        # Convert to image
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return data
    
    def predict(self, image: Image.Image, model_type: str, category: str) -> Dict:
        """
        Run inference on an image.
        
        Args:
            image: PIL Image
            model_type: CAE, VAE, or DAE
            category: MVTec category
            
        Returns:
            Dict with original, reconstruction, heatmap (as base64), and score
        """
        # Load model
        model = self._get_cached_model(model_type, category)
        
        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            if model_type.upper() == "VAE":
                reconstruction, _, _ = model(input_tensor)
                anomaly_score = model.get_anomaly_score(input_tensor).item()
                error_map = model.get_anomaly_map(input_tensor)[0, 0].cpu().numpy()
            else:
                reconstruction = model(input_tensor)
                anomaly_score = model.get_reconstruction_error(input_tensor, reduction='mean').item()
                error_map = model.get_anomaly_map(input_tensor)[0, 0].cpu().numpy()
        
        # Convert to numpy images
        original_np = denormalize(input_tensor[0].cpu()).permute(1, 2, 0).numpy().clip(0, 1)
        recon_np = reconstruction[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        
        # Create heatmap overlay
        heatmap = self._create_heatmap(error_map, original_np)
        
        return {
            "anomaly_score": float(anomaly_score),
            "original_base64": self._image_to_base64(original_np),
            "reconstruction_base64": self._image_to_base64(recon_np),
            "heatmap_base64": self._image_to_base64(heatmap),
        }
    
    def get_available_models(self) -> List[Dict]:
        """Get list of available trained models."""
        models = []
        for model_type, prefix in [("CAE", "cae"), ("VAE", "vae"), ("DAE", "dae")]:
            for category in MVTEC_CATEGORIES:
                path = self.models_dir / f"{prefix}_{category}_final.pth"
                if path.exists():
                    models.append({
                        "name": f"{model_type} - {category}",
                        "type": model_type,
                        "category": category,
                        "file_size_mb": round(path.stat().st_size / 1024 / 1024, 2),
                    })
        return models
    
    def get_available_categories(self) -> List[str]:
        """Get categories with at least one trained model."""
        categories = set()
        for prefix in ["cae", "vae", "dae"]:
            for category in MVTEC_CATEGORIES:
                path = self.models_dir / f"{prefix}_{category}_final.pth"
                if path.exists():
                    categories.add(category)
        return sorted(list(categories))
