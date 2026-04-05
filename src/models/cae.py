"""
Convolutional Autoencoder (CAE) for Anomaly Detection.

The CAE learns to reconstruct normal images. During inference,
defective regions produce high reconstruction error.

Architecture:
    Encoder: Input -> Conv blocks -> Latent space
    Decoder: Latent space -> Transposed Conv blocks -> Reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class ConvBlock(nn.Module):
    """Convolutional block with Conv2d, BatchNorm, and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DeconvBlock(nn.Module):
    """Transposed convolutional block for upsampling."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size,
                stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


class CAEEncoder(nn.Module):
    """Encoder network for CAE."""
    
    def __init__(self, in_channels: int = 3, channels: List[int] = [32, 64, 128, 256]):
        super().__init__()
        
        layers = []
        prev_channels = in_channels
        
        for ch in channels:
            layers.append(ConvBlock(prev_channels, ch))
            prev_channels = ch
        
        self.encoder = nn.Sequential(*layers)
        self.out_channels = channels[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CAEDecoder(nn.Module):
    """Decoder network for CAE."""
    
    def __init__(self, out_channels: int = 3, channels: List[int] = [256, 128, 64, 32]):
        super().__init__()
        
        layers = []
        for i in range(len(channels) - 1):
            layers.append(DeconvBlock(channels[i], channels[i + 1]))
        
        # Final layer without BatchNorm or activation (linear output)
        layers.append(nn.ConvTranspose2d(
            channels[-1], out_channels, 3, stride=2, padding=1, output_padding=1
        ))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for anomaly detection.
    
    The model learns to reconstruct normal industrial images.
    Anomalies are detected by computing reconstruction error.
    
    Args:
        in_channels: Number of input channels (3 for RGB)
        channels: List of channel sizes for encoder layers
        
    Input shape: (B, 3, 256, 256)
    Output shape: (B, 3, 256, 256)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [32, 64, 128, 256],
    ):
        super().__init__()
        
        self.encoder = CAEEncoder(in_channels, channels)
        self.decoder = CAEDecoder(in_channels, channels[::-1])
        
        # Store architecture info
        self.in_channels = in_channels
        self.channels = channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and decoder."""
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
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
            recon = self.forward(x)
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
            Anomaly maps (B, 1, H, W) - higher values indicate anomalies
        """
        error = self.get_reconstruction_error(x, reduction='none')
        # Average across channels for single-channel anomaly map
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
    
    def _compute_ssim_map(
        self, x: torch.Tensor, recon: torch.Tensor,
        window_size: int = 11, channels: int = 3,
    ) -> torch.Tensor:
        """
        Compute SSIM map between original and reconstruction.
        
        Returns:
            SSIM dissimilarity map (B, 1, H, W) — higher = more anomalous
        """
        # Gaussian window
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        window_1d = gauss.unsqueeze(1)
        window_2d = window_1d @ window_1d.t()
        window = window_2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1).to(x.device)
        
        pad = window_size // 2
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = F.conv2d(x, window, padding=pad, groups=channels)
        mu2 = F.conv2d(recon, window, padding=pad, groups=channels)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(x ** 2, window, padding=pad, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(recon ** 2, window, padding=pad, groups=channels) - mu2_sq
        sigma12 = F.conv2d(x * recon, window, padding=pad, groups=channels) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # Average across channels, return dissimilarity
        return (1 - ssim_map).mean(dim=1, keepdim=True)
    
    def get_anomaly_map_ssim(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate SSIM-based anomaly map (captures structural anomalies).
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Anomaly maps (B, 1, H, W)
        """
        with torch.no_grad():
            recon = self.forward(x)
            return self._compute_ssim_map(x, recon)
    
    def get_anomaly_map_combined(
        self, x: torch.Tensor, ssim_weight: float = 0.5
    ) -> torch.Tensor:
        """
        Generate combined MSE + SSIM anomaly map.
        
        Combines pixel-wise MSE error with structural SSIM dissimilarity
        for more robust anomaly detection.
        
        Args:
            x: Input images (B, C, H, W)
            ssim_weight: Weight for SSIM component (0-1)
            
        Returns:
            Combined anomaly maps (B, 1, H, W)
        """
        with torch.no_grad():
            recon = self.forward(x)
            mse_map = ((x - recon) ** 2).mean(dim=1, keepdim=True)
            ssim_map = self._compute_ssim_map(x, recon)
            
            # Normalize both to [0, 1] range for fair combination
            mse_norm = mse_map / (mse_map.amax(dim=(2, 3), keepdim=True) + 1e-8)
            ssim_norm = ssim_map / (ssim_map.amax(dim=(2, 3), keepdim=True) + 1e-8)
            
            return (1 - ssim_weight) * mse_norm + ssim_weight * ssim_norm


def create_cae(
    channels: List[int] = [32, 64, 128, 256],
    pretrained: bool = False,
) -> ConvAutoencoder:
    """
    Factory function to create CAE model.
    
    Args:
        channels: Channel sizes for encoder layers
        pretrained: If True, load pretrained weights (not implemented)
        
    Returns:
        ConvAutoencoder model
    """
    model = ConvAutoencoder(in_channels=3, channels=channels)
    
    if pretrained:
        raise NotImplementedError("Pretrained weights not available")
    
    return model


if __name__ == "__main__":
    # Quick test
    model = create_cae()
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
