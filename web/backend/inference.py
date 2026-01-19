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
from src.models.cnn_classifier import create_cnn_classifier
from src.data.transforms import get_transforms, denormalize

# NEU Surface Defect class names
NEU_CLASSES = ["Crazing", "Inclusion", "Patches", "Pitted", "Rolled", "Scratches"]


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
        prefix_map = {
            "CAE": "cae",
            "VAE": "vae",
            "DAE": "dae",
        }
        prefix = prefix_map.get(model_type.upper(), "cae")
        return self.models_dir / f"{prefix}_{category}_final.pth"
    
    def _load_model(self, model_type: str, category: str = None) -> torch.nn.Module:
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
        elif model_type.upper() == "CNN":
            model = create_cnn_classifier(num_classes=6)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.to(self.device)
        
        return model
    
    def _get_cached_model(self, model_type: str, category: str = None) -> torch.nn.Module:
        """Get model from cache or load it."""
        if model_type.upper() == "CNN":
            key = "CNN"
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
        
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return data
    
    def predict(self, image: Image.Image, model_type: str, category: str = None) -> Dict:
        """
        Run inference on an image.
        
        Args:
            image: PIL Image
            model_type: CAE, VAE, DAE, or CNN
            category: MVTec category (not needed for CNN)
            
        Returns:
            Dict with results (format depends on model type)
        """
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
            
            elif model_type.upper() == "VAE":
                reconstruction, _, _ = model(input_tensor)
                anomaly_score = model.get_anomaly_score(input_tensor).item()
                error_map = model.get_anomaly_map(input_tensor)[0, 0].cpu().numpy()
            elif model_type.upper() == "DAE":
                # DAE forward() returns tuple, use reconstruct() for direct reconstruction
                reconstruction = model.reconstruct(input_tensor)
                anomaly_score = model.get_reconstruction_error(input_tensor, reduction='mean').item()
                error_map = model.get_anomaly_map(input_tensor)[0, 0].cpu().numpy()
            else:
                # CAE
                reconstruction = model(input_tensor)
                anomaly_score = model.get_reconstruction_error(input_tensor, reduction='mean').item()
                error_map = model.get_anomaly_map(input_tensor)[0, 0].cpu().numpy()
        
        # Convert to numpy images
        original_np = denormalize(input_tensor[0].cpu()).permute(1, 2, 0).numpy().clip(0, 1)
        recon_np = reconstruction[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        
        # Create heatmap overlay
        heatmap = self._create_heatmap(error_map, original_np)
        
        return {
            "model_type": model_type.upper(),
            "is_classifier": False,
            "anomaly_score": float(anomaly_score),
            "original_base64": self._image_to_base64(original_np),
            "reconstruction_base64": self._image_to_base64(recon_np),
            "heatmap_base64": self._image_to_base64(heatmap),
        }
    
    def get_available_models(self) -> List[Dict]:
        """Get list of available trained models."""
        models = []
        
        # Check autoencoders
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
        for prefix in ["cae", "vae", "dae"]:
            for category in MVTEC_CATEGORIES:
                path = self.models_dir / f"{prefix}_{category}_final.pth"
                if path.exists():
                    categories.add(category)
        return sorted(list(categories))
    
    def is_cnn_available(self) -> bool:
        """Check if CNN classifier model is available."""
        return (self.models_dir / "cnn_classifier_final.pth").exists()

