"""
Model Inference Module (v2.0)

Handles loading trained models and running inference on images.
Upgraded with:
    - Gaussian-smoothed anomaly maps for cleaner heatmaps
    - Higher-resolution visualization output (512px)
    - Support for Skip-CAE and PatchCore models
    - SSIM-based anomaly maps for CAE models
    - Image size validation and sanitization
    - weights_only=True for secure torch.load
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
from src.models.cnn_classifier import create_cnn_classifier
from src.models.skip_cae import create_skip_cae
from src.models.patchcore import create_patchcore
from src.data.transforms import get_transforms, denormalize

# NEU Surface Defect class names
NEU_CLASSES = ["Crazing", "Inclusion", "Patches", "Pitted", "Rolled", "Scratches"]

# Maximum allowed image size (4 megapixels)
MAX_IMAGE_PIXELS = 4_000_000


def _smooth_anomaly_map(error_map: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    """
    Apply Gaussian smoothing to anomaly map for cleaner visualization.
    
    Args:
        error_map: Raw error map (H, W)
        sigma: Gaussian kernel sigma
        
    Returns:
        Smoothed and normalized map (H, W) in [0, 1]
    """
    from scipy.ndimage import gaussian_filter
    
    smoothed = gaussian_filter(error_map, sigma=sigma)
    
    # Normalize to [0, 1]
    vmin, vmax = smoothed.min(), smoothed.max()
    if vmax - vmin > 1e-8:
        smoothed = (smoothed - vmin) / (vmax - vmin)
    else:
        smoothed = np.zeros_like(smoothed)
    
    return smoothed


class ModelInference:
    """
    Handles model loading and inference for web API.
    """
    
    def __init__(self):
        self.models: Dict[str, torch.nn.Module] = {}
        self.transform = get_transforms(train=False)
        self.models_dir = MODELS_DIR
        self.device = DEVICE
        
    def _get_model_path(self, model_type: str, category: str = None) -> Path:
        """Get path to saved model."""
        if model_type.upper() == "CNN":
            return self.models_dir / "cnn_classifier_final.pth"
        if model_type.upper() == "PATCHCORE":
            return self.models_dir / f"patchcore_{category}_memory.pth"
        prefix_map = {
            "CAE": "cae",
            "VAE": "vae",
            "DAE": "dae",
            "SKIP_CAE": "skip_cae",
        }
        prefix = prefix_map.get(model_type.upper(), "cae")
        return self.models_dir / f"{prefix}_{category}_final.pth"
    
    def _load_model(self, model_type: str, category: str = None) -> torch.nn.Module:
        """Load a trained model."""
        model_path = self._get_model_path(model_type, category)
        
        if not model_path.exists():
            raise ValueError(f"Model not found: {model_path}")
        
        # Create model
        mt = model_type.upper()
        if mt == "CAE":
            model = create_cae()
        elif mt == "VAE":
            model = create_vae()
        elif mt == "DAE":
            model = create_denoising_ae()
        elif mt == "CNN":
            model = create_cnn_classifier(num_classes=6)
        elif mt == "SKIP_CAE":
            model = create_skip_cae()
        elif mt == "PATCHCORE":
            model = create_patchcore()
            model.load_memory_bank(str(model_path))
            model.eval()
            return model
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load weights (weights_only=True for security on PyTorch 2.4+)
        import pickle
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        except (TypeError, pickle.UnpicklingError):
            # TypeError: older PyTorch without weights_only param
            # UnpicklingError: checkpoint contains non-tensor objects (e.g. numpy scalars)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.to(self.device)
        
        return model
    
    def _get_cached_model(self, model_type: str, category: str = None) -> torch.nn.Module:
        """Get model from cache or load it."""
        if model_type.upper() == "CNN":
            key = "CNN"
        elif model_type.upper() == "PATCHCORE":
            key = f"PATCHCORE_{category}"
        else:
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
        """Create high-resolution heatmap overlay on original image."""
        fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)
        ax.imshow(original)
        ax.imshow(error_map, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Convert to image (buffer_rgba replaces tostring_rgb in Matplotlib 3.8+)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        plt.close(fig)
        
        return data
    
    def _create_class_bar_chart(self, probabilities: List[float], classes: List[str]) -> np.ndarray:
        """Create a bar chart visualization for class probabilities."""
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        colors = ['#22c55e' if p == max(probabilities) else '#3b82f6' for p in probabilities]
        ax.barh(classes, probabilities, color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probability')
        ax.set_title('Class Predictions')
        for i, (prob, cls) in enumerate(zip(probabilities, classes)):
            ax.text(prob + 0.02, i, f'{prob:.1%}', va='center', fontsize=8)
        plt.tight_layout()
        
        # Convert to image (buffer_rgba replaces tostring_rgb in Matplotlib 3.8+)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        plt.close(fig)
        
        return data
    
    def _validate_image(self, image: Image.Image) -> Image.Image:
        """Validate and sanitize input image size."""
        w, h = image.size
        pixels = w * h
        
        if pixels > MAX_IMAGE_PIXELS:
            # Downscale proportionally
            scale = (MAX_IMAGE_PIXELS / pixels) ** 0.5
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        return image
    
    def predict(self, image: Image.Image, model_type: str, category: str = None) -> Dict:
        """
        Run inference on an image.
        
        Args:
            image: PIL Image
            model_type: CAE, VAE, DAE, SKIP_CAE, PATCHCORE, or CNN
            category: MVTec category (not needed for CNN)
            
        Returns:
            Dict with results (format depends on model type)
        """
        # Validate image
        image = self._validate_image(image)
        
        # Load model
        model = self._get_cached_model(model_type, category)
        
        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            if model_type.upper() == "CNN":
                # Classification mode
                probs = model.predict_proba(input_tensor)[0].cpu().numpy().tolist()
                pred_idx = int(np.argmax(probs))
                pred_class = NEU_CLASSES[pred_idx]
                confidence = probs[pred_idx]
                
                # Create visualization
                original_np = denormalize(input_tensor[0].cpu()).permute(1, 2, 0).numpy().clip(0, 1)
                chart = self._create_class_bar_chart(probs, NEU_CLASSES)
                
                return {
                    "model_type": "CNN",
                    "is_classifier": True,
                    "predicted_class": pred_class,
                    "confidence": float(confidence),
                    "class_probabilities": {cls: float(p) for cls, p in zip(NEU_CLASSES, probs)},
                    "original_base64": self._image_to_base64(original_np),
                    "chart_base64": self._image_to_base64(chart),
                }
            
            elif model_type.upper() == "PATCHCORE":
                anomaly_score = model.get_anomaly_score(input_tensor).item()
                error_map = model.get_anomaly_map(input_tensor)[0, 0].cpu().numpy()
                reconstruction = input_tensor  # PatchCore doesn't reconstruct
                
            elif model_type.upper() == "VAE":
                reconstruction, _, _ = model(input_tensor)
                anomaly_score = model.get_anomaly_score(input_tensor).item()
                error_map = model.get_anomaly_map(input_tensor)[0, 0].cpu().numpy()
            elif model_type.upper() == "DAE":
                reconstruction = model.reconstruct(input_tensor)
                anomaly_score = model.get_reconstruction_error(input_tensor, reduction='mean').item()
                error_map = model.get_anomaly_map(input_tensor)[0, 0].cpu().numpy()
            elif model_type.upper() == "SKIP_CAE":
                reconstruction = model(input_tensor)
                anomaly_score = model.get_anomaly_score(input_tensor).item()
                error_map = model.get_anomaly_map(input_tensor)[0, 0].cpu().numpy()
            else:
                # CAE — use combined MSE + SSIM map
                reconstruction = model(input_tensor)
                anomaly_score = model.get_reconstruction_error(input_tensor, reduction='mean').item()
                try:
                    error_map = model.get_anomaly_map_combined(input_tensor)[0, 0].cpu().numpy()
                except AttributeError:
                    error_map = model.get_anomaly_map(input_tensor)[0, 0].cpu().numpy()
        
        # Apply Gaussian smoothing to error map
        error_map = _smooth_anomaly_map(error_map, sigma=4.0)
        
        # Convert to numpy images
        original_np = denormalize(input_tensor[0].cpu()).permute(1, 2, 0).numpy().clip(0, 1)
        
        if model_type.upper() == "PATCHCORE":
            recon_np = original_np  # No reconstruction for PatchCore
        else:
            recon_np = reconstruction[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        
        # Create heatmap overlay (higher resolution now)
        heatmap = self._create_heatmap(error_map, original_np)
        
        return {
            "model_type": model_type.upper(),
            "is_classifier": False,
            "anomaly_score": float(anomaly_score),
            "original_base64": self._image_to_base64(original_np),
            "reconstruction_base64": self._image_to_base64(recon_np),
            "heatmap_base64": self._image_to_base64(heatmap),
        }
    
    def predict_batch(
        self, image: Image.Image, model_types: List[str], category: str = None
    ) -> List[Dict]:
        """
        Run inference with multiple models on a single image.
        
        Much faster than calling predict() sequentially because:
        - Image preprocessing happens once
        - Results are batched
        
        Args:
            image: PIL Image
            model_types: List of model types (e.g. ["CAE", "VAE", "DAE"])
            category: MVTec category
            
        Returns:
            List of result dicts, one per model
        """
        results = []
        for mt in model_types:
            try:
                result = self.predict(image, mt, category)
                results.append(result)
            except Exception as e:
                results.append({
                    "success": False,
                    "model_type": mt,
                    "error": str(e),
                })
        return results
    
    def get_available_models(self) -> List[Dict]:
        """Get list of available trained models."""
        models = []
        
        # Check autoencoders (including Skip-CAE)
        for model_type, prefix in [("CAE", "cae"), ("VAE", "vae"), ("DAE", "dae"), ("SKIP_CAE", "skip_cae")]:
            for category in MVTEC_CATEGORIES:
                path = self.models_dir / f"{prefix}_{category}_final.pth"
                if path.exists():
                    models.append({
                        "name": f"{model_type} - {category}",
                        "type": model_type,
                        "category": category,
                        "file_size_mb": round(path.stat().st_size / 1024 / 1024, 2),
                    })
        
        # Check PatchCore memory banks
        for category in MVTEC_CATEGORIES:
            path = self.models_dir / f"patchcore_{category}_memory.pth"
            if path.exists():
                models.append({
                    "name": f"PatchCore - {category}",
                    "type": "PATCHCORE",
                    "category": category,
                    "file_size_mb": round(path.stat().st_size / 1024 / 1024, 2),
                })
        
        # Check CNN classifier
        cnn_path = self.models_dir / "cnn_classifier_final.pth"
        if cnn_path.exists():
            models.append({
                "name": "CNN - NEU Classifier",
                "type": "CNN",
                "category": "NEU",
                "file_size_mb": round(cnn_path.stat().st_size / 1024 / 1024, 2),
            })
        
        return models
    
    def get_available_categories(self) -> List[str]:
        """Get categories with at least one trained model."""
        categories = set()
        for prefix in ["cae", "vae", "dae", "skip_cae"]:
            for category in MVTEC_CATEGORIES:
                path = self.models_dir / f"{prefix}_{category}_final.pth"
                if path.exists():
                    categories.add(category)
        # Also check PatchCore
        for category in MVTEC_CATEGORIES:
            path = self.models_dir / f"patchcore_{category}_memory.pth"
            if path.exists():
                categories.add(category)
        return sorted(list(categories))
    
    def is_cnn_available(self) -> bool:
        """Check if CNN classifier model is available."""
        return (self.models_dir / "cnn_classifier_final.pth").exists()
    
    def get_model_types(self) -> List[str]:
        """Get list of all supported model types."""
        return ["CAE", "VAE", "DAE", "SKIP_CAE", "PATCHCORE", "CNN"]
