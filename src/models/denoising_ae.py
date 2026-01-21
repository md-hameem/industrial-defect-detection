"""
Denoising Autoencoder for Anomaly Detection.

The Denoising AE is trained to reconstruct clean images from
noisy inputs, forcing the model to learn robust features.
This approach often improves anomaly detection performance.

Architecture:
    Same as CAE, but with noise injection during training.
"""

import torch
import torch.nn as nn
from typing import List, Tuple

from src.models.cae import ConvAutoencoder, create_cae


class DenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder for anomaly detection.
    
    During training, Gaussian noise is added to input images.
    The model learns to reconstruct the clean original image.
    This makes the model more robust to input variations.
    
    Args:
        in_channels: Number of input channels (3 for RGB)
        channels: List of channel sizes for encoder layers
        noise_factor: Standard deviation of noise to add (0.0 to 1.0)
        
    Input shape: (B, 3, 256, 256)
    Output shape: (B, 3, 256, 256)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [32, 64, 128, 256],
        noise_factor: float = 0.3,
    ):
        super().__init__()
        
        # Use the same architecture as CAE
        self.autoencoder = ConvAutoencoder(in_channels, channels)
        self.noise_factor = noise_factor
    
    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to input tensor.
        
        Args:
            x: Clean input tensor
            
        Returns:
            Noisy tensor (clamped to [0, 1])
        """
        noise = torch.randn_like(x) * self.noise_factor
        noisy = x + noise
        return noisy  # Do not clamp for normalized data
    
    def forward(
        self, x: torch.Tensor, add_noise: bool = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional noise injection.
        
        Args:
            x: Input images (clean)
            add_noise: Whether to add noise (default: True if training)
            
        Returns:
            Tuple of (reconstruction, noisy_input)
            If not adding noise, noisy_input equals x
        """
        if add_noise is None:
            add_noise = self.training
        
        if add_noise:
            noisy_x = self.add_noise(x)
        else:
            noisy_x = x
        
        reconstruction = self.autoencoder(noisy_x)
        return reconstruction, noisy_x
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.autoencoder.encode(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.autoencoder.decode(z)
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct without adding noise (for inference).
        
        Args:
            x: Input images
            
        Returns:
            Reconstructed images
        """
        return self.autoencoder(x)
    
    def get_reconstruction_error(
        self, x: torch.Tensor, reduction: str = 'none'
    ) -> torch.Tensor:
        """
        Compute reconstruction error for anomaly detection.
        
        Args:
            x: Input images
            reduction: 'none' for pixel-wise, 'mean' for image-level
            
        Returns:
            Reconstruction error map or scalar
        """
        with torch.no_grad():
            recon = self.reconstruct(x)
            error = (x - recon) ** 2
            
            if reduction == 'none':
                return error
            elif reduction == 'mean':
                return error.mean(dim=[1, 2, 3])
            else:
                raise ValueError(f"Unknown reduction: {reduction}")
    
    def get_anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate pixel-wise anomaly map.
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Anomaly maps (B, 1, H, W)
        """
        error = self.get_reconstruction_error(x, reduction='none')
        anomaly_map = error.mean(dim=1, keepdim=True)
        return anomaly_map
    
    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get image-level anomaly score.
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Anomaly scores (B,) - higher values indicate anomalies
        """
        return self.get_reconstruction_error(x, reduction='mean')


def create_denoising_ae(
    channels: List[int] = [32, 64, 128, 256],
    noise_factor: float = 0.3,
) -> DenoisingAutoencoder:
    """
    Factory function to create Denoising AE model.
    
    Args:
        channels: Channel sizes for encoder layers
        noise_factor: Standard deviation of noise (0.0 to 1.0)
        
    Returns:
        DenoisingAutoencoder model
    """
    return DenoisingAutoencoder(
        in_channels=3,
        channels=channels,
        noise_factor=noise_factor,
    )


if __name__ == "__main__":
    # Quick test
    model = create_denoising_ae(noise_factor=0.3)
    x = torch.randn(2, 3, 256, 256).clamp(0, 1)
    
    # Training mode (with noise)
    model.train()
    recon, noisy = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Noisy input shape: {noisy.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Noise factor: {model.noise_factor}")
    
    # Eval mode (without noise)
    model.eval()
    recon_clean, _ = model(x)
    print(f"Clean reconstruction shape: {recon_clean.shape}")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
