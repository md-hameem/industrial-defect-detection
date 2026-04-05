"""
Skip-Connection Convolutional Autoencoder (U-Net Style) for Anomaly Detection.

Adds encoder-to-decoder skip connections to preserve spatial detail.
This dramatically improves pixel-level anomaly maps compared to vanilla CAE,
because high-resolution features bypass the bottleneck and are directly
available during reconstruction.

Key Difference vs CAE:
    - CAE bottleneck: 256 x 16 x 16 = 65,536 dims (spatial but low-res)
    - Skip-CAE: Preserves full-res features via skip connections
    - Result: Sharper reconstructions, more precise anomaly localization

Architecture:
    Encoder: Input -> Conv blocks -> (skip connections stored) -> Bottleneck
    Decoder: Bottleneck -> (concat skip connections) -> TransConv blocks -> Output
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class SkipEncoderBlock(nn.Module):
    """Encoder block that outputs both downsampled and skip features."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple of (downsampled, skip_features)
        """
        x = self.relu(self.bn1(self.conv1(x)))
        skip = self.relu(self.bn2(self.conv2(x)))
        down = self.pool(skip)
        return down, skip


class SkipDecoderBlock(nn.Module):
    """Decoder block that accepts skip connection from encoder."""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        # After concatenation: out_channels + skip_channels
        self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch from pooling
        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class SkipConvAutoencoder(nn.Module):
    """
    U-Net style Convolutional Autoencoder with skip connections.
    
    Skip connections pass high-resolution features from encoder to decoder,
    producing sharper reconstructions and more accurate anomaly maps.
    
    For anomaly detection, defects cause mismatches in skip connections
    (trained on normal data only), making them even more visible in
    the error map.
    
    Args:
        in_channels: Number of input channels (3 for RGB)
        channels: List of channel sizes for encoder levels
        
    Input shape: (B, 3, 256, 256)
    Output shape: (B, 3, 256, 256)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        channels: List[int] = [32, 64, 128, 256],
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.channels = channels
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels:
            self.encoder_blocks.append(SkipEncoderBlock(prev_ch, ch))
            prev_ch = ch
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1] * 2, 3, padding=1),
            nn.BatchNorm2d(channels[-1] * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[-1] * 2, channels[-1], 3, padding=1),
            nn.BatchNorm2d(channels[-1]),
            nn.ReLU(inplace=True),
        )
        
        # Decoder blocks (reverse order)
        self.decoder_blocks = nn.ModuleList()
        reversed_channels = channels[::-1]
        for i in range(len(reversed_channels) - 1):
            self.decoder_blocks.append(
                SkipDecoderBlock(
                    in_channels=reversed_channels[i],
                    skip_channels=reversed_channels[i + 1],
                    out_channels=reversed_channels[i + 1],
                )
            )
        
        # Final upsampling + output
        self.final_up = nn.ConvTranspose2d(channels[0], channels[0], 2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels[0] + in_channels, channels[0], 3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[0], in_channels, 1),  # 1x1 conv to output channels
        )
        
        # Store input for skip connection
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder, bottleneck, and decoder with skips."""
        # Store input for final skip
        input_skip = self.input_conv(x)
        
        # Encoder
        skips = []
        h = x
        for encoder_block in self.encoder_blocks:
            h, skip = encoder_block(h)
            skips.append(skip)
        
        # Bottleneck
        h = self.bottleneck(h)
        
        # Decoder (skips are in reverse order)
        skips = skips[::-1]
        for i, decoder_block in enumerate(self.decoder_blocks):
            h = decoder_block(h, skips[i + 1])
        
        # Final upsampling with input skip
        h = self.final_up(h)
        if h.shape != input_skip.shape:
            h = nn.functional.interpolate(h, size=input_skip.shape[2:], mode='bilinear', align_corners=False)
        h = torch.cat([h, input_skip], dim=1)
        reconstruction = self.final_conv(h)
        
        return reconstruction
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to bottleneck representation."""
        h = x
        for encoder_block in self.encoder_blocks:
            h, _ = encoder_block(h)
        return self.bottleneck(h)
    
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


def create_skip_cae(
    channels: List[int] = [32, 64, 128, 256],
) -> SkipConvAutoencoder:
    """
    Factory function to create Skip-Connection CAE.
    
    Args:
        channels: Channel sizes for encoder levels
        
    Returns:
        SkipConvAutoencoder model
    """
    return SkipConvAutoencoder(in_channels=3, channels=channels)


if __name__ == "__main__":
    # Quick test
    model = create_skip_cae()
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test error computation
    error = model.get_reconstruction_error(x, reduction='mean')
    print(f"Anomaly scores: {error}")
    
    anomaly_map = model.get_anomaly_map(x)
    print(f"Anomaly map shape: {anomaly_map.shape}")
