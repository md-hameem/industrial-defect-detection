"""
Loss functions for anomaly detection models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MSELoss(nn.Module):
    """Mean Squared Error loss for autoencoders."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(input, target, reduction=self.reduction)


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index loss.
    
    Better captures perceptual similarity than MSE.
    Loss = 1 - SSIM (so minimizing loss maximizes SSIM)
    """
    
    def __init__(self, window_size: int = 11, channels: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        
        # Create Gaussian window
        self.register_buffer('window', self._create_window(window_size, channels))
    
    def _create_window(self, window_size: int, channels: int) -> torch.Tensor:
        """Create Gaussian window for SSIM computation."""
        sigma = 1.5
        gauss = torch.tensor([
            torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)))
            for x in range(window_size)
        ])
        gauss = gauss / gauss.sum()
        
        window_1d = gauss.unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t()).unsqueeze(0).unsqueeze(0)
        window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
        
        return window
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        window = self.window
        
        mu1 = F.conv2d(input, window, padding=self.window_size // 2, groups=self.channels)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=self.channels)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(input ** 2, window, padding=self.window_size // 2, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(target ** 2, window, padding=self.window_size // 2, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(input * target, window, padding=self.window_size // 2, groups=self.channels) - mu1_mu2
        
        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim.mean()


class CombinedLoss(nn.Module):
    """
    Combined MSE and SSIM loss.
    
    Loss = alpha * MSE + (1 - alpha) * SSIM_loss
    """
    
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = MSELoss()
        self.ssim = SSIMLoss()
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(input, target)
        ssim_loss = self.ssim(input, target)
        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss


class VAELoss(nn.Module):
    """
    VAE loss combining reconstruction and KL divergence.
    
    Loss = MSE(input, recon) + beta * KL(q(z|x) || p(z))
    """
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
    
    def forward(
        self,
        input: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, input, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + self.beta * kl_loss


class DenoisingLoss(nn.Module):
    """
    Loss for denoising autoencoder.
    
    Compares reconstruction to the CLEAN (not noisy) input.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
    
    def forward(
        self,
        clean_input: torch.Tensor,
        reconstruction: torch.Tensor,
        noisy_input: torch.Tensor = None,  # Not used, for API compatibility
    ) -> torch.Tensor:
        """Compare reconstruction to clean input."""
        return self.mse(reconstruction, clean_input)


def get_loss_function(model_type: str, **kwargs) -> nn.Module:
    """
    Factory function to get appropriate loss for model type.
    
    Args:
        model_type: 'cae', 'vae', 'denoising', or 'classifier'
        **kwargs: Additional loss arguments
        
    Returns:
        Loss function module
    """
    if model_type == 'cae':
        return CombinedLoss(alpha=kwargs.get('alpha', 0.8))
    elif model_type == 'vae':
        return VAELoss(beta=kwargs.get('beta', 1.0))
    elif model_type == 'denoising':
        return DenoisingLoss()
    elif model_type == 'classifier':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
