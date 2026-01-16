"""
Variational Autoencoder (VAE) for Anomaly Detection.

The VAE learns a probabilistic latent space representation.
Anomalies are detected using both reconstruction error and
KL divergence from the learned prior.

Architecture:
    Encoder: Input -> Conv blocks -> Mean & LogVar
    Reparameterization: z = mean + std * epsilon
    Decoder: z -> Transposed Conv blocks -> Reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict


class VAEEncoder(nn.Module):
    """Encoder network for VAE with mean and log-variance outputs."""
    
    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [32, 64, 128, 256],
        latent_dim: int = 128,
    ):
        super().__init__()
        
        # Convolutional layers
        layers = []
        prev_channels = in_channels
        for ch in channels:
            layers.extend([
                nn.Conv2d(prev_channels, ch, 3, stride=2, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ])
            prev_channels = ch
        
        self.conv = nn.Sequential(*layers)
        
        # Calculate flattened size after convolutions
        # For 256x256 input with 4 conv layers (each halves): 256 / 2^4 = 16
        self.flat_size = channels[-1] * 16 * 16
        
        # Fully connected layers for mean and log-variance
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to mean and log-variance.
        
        Returns:
            Tuple of (mean, log_variance)
        """
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class VAEDecoder(nn.Module):
    """Decoder network for VAE."""
    
    def __init__(
        self,
        out_channels: int = 3,
        channels: List[int] = [256, 128, 64, 32],
        latent_dim: int = 128,
    ):
        super().__init__()
        
        self.channels = channels
        
        # Project latent to spatial representation
        self.fc = nn.Linear(latent_dim, channels[0] * 16 * 16)
        
        # Transposed convolutional layers
        layers = []
        for i in range(len(channels) - 1):
            layers.extend([
                nn.ConvTranspose2d(
                    channels[i], channels[i + 1], 3,
                    stride=2, padding=1, output_padding=1
                ),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(inplace=True),
            ])
        
        # Final layer without activation (linear output)
        layers.extend([
            nn.ConvTranspose2d(
                channels[-1], out_channels, 3,
                stride=2, padding=1, output_padding=1
            ),
        ])
        
        self.deconv = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        h = self.fc(z)
        h = h.view(h.size(0), self.channels[0], 16, 16)
        return self.deconv(h)


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder for anomaly detection.
    
    Uses the reparameterization trick for end-to-end training.
    Loss = Reconstruction Loss + beta * KL Divergence
    
    Args:
        in_channels: Number of input channels (3 for RGB)
        channels: List of channel sizes for encoder layers
        latent_dim: Dimension of latent space
        beta: Weight for KL divergence term (beta-VAE)
        
    Input shape: (B, 3, 256, 256)
    Output shape: (B, 3, 256, 256)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [32, 64, 128, 256],
        latent_dim: int = 128,
        beta: float = 1.0,
    ):
        super().__init__()
        
        self.encoder = VAEEncoder(in_channels, channels, latent_dim)
        self.decoder = VAEDecoder(in_channels, channels[::-1], latent_dim)
        
        self.latent_dim = latent_dim
        self.beta = beta
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            # During inference, use mean directly
            return mu
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Returns:
            Tuple of (reconstruction, mean, log_variance)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation (mean)."""
        mu, _ = self.encoder(x)
        return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z)
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample from the learned latent prior.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            Generated images
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
    
    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence from standard normal.
        
        KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        """
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    
    def loss_function(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss.
        
        Args:
            x: Original input
            recon: Reconstruction
            mu: Latent mean
            logvar: Latent log variance
            
        Returns:
            Dictionary with total loss and component losses
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, x, reduction='mean')
        
        # KL divergence
        kl_loss = self.kl_divergence(mu, logvar).mean()
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
        }
    
    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly score combining reconstruction and KL.
        
        Args:
            x: Input images
            
        Returns:
            Anomaly scores for each image
        """
        with torch.no_grad():
            recon, mu, logvar = self.forward(x)
            
            # Reconstruction error
            recon_error = ((x - recon) ** 2).mean(dim=[1, 2, 3])
            
            # KL divergence
            kl = self.kl_divergence(mu, logvar)
            
            # Combined score
            score = recon_error + self.beta * kl
            
        return score
    
    def get_anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate pixel-wise anomaly map.
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Anomaly maps (B, 1, H, W)
        """
        with torch.no_grad():
            recon, _, _ = self.forward(x)
            error = ((x - recon) ** 2).mean(dim=1, keepdim=True)
        return error


def create_vae(
    channels: List[int] = [32, 64, 128, 256],
    latent_dim: int = 128,
    beta: float = 1.0,
) -> VariationalAutoencoder:
    """
    Factory function to create VAE model.
    
    Args:
        channels: Channel sizes for encoder layers
        latent_dim: Dimension of latent space
        beta: Weight for KL divergence (beta-VAE)
        
    Returns:
        VariationalAutoencoder model
    """
    return VariationalAutoencoder(
        in_channels=3,
        channels=channels,
        latent_dim=latent_dim,
        beta=beta,
    )


if __name__ == "__main__":
    # Quick test
    model = create_vae()
    x = torch.randn(2, 3, 256, 256)
    recon, mu, logvar = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Latent shape: {mu.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test loss computation
    losses = model.loss_function(x, recon, mu, logvar)
    print(f"Total loss: {losses['loss']:.4f}")
    print(f"Recon loss: {losses['recon_loss']:.4f}")
    print(f"KL loss: {losses['kl_loss']:.4f}")
