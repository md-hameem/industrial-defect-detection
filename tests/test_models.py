"""
Unit tests for model architectures.

Tests all 6 models: CAE, VAE, DAE, SkipCAE, PatchCore, CNN.
Verifies forward pass, output shapes, anomaly scores, and anomaly maps.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# CAE Tests
# =============================================================================

class TestCAE:
    """Tests for Convolutional Autoencoder."""
    
    def test_forward_shape(self):
        from src.models.cae import create_cae
        model = create_cae()
        x = torch.randn(2, 3, 256, 256)
        out = model(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    
    def test_encode_shape(self):
        from src.models.cae import create_cae
        model = create_cae()
        x = torch.randn(2, 3, 256, 256)
        latent = model.encode(x)
        assert latent.shape == (2, 256, 16, 16), f"Unexpected latent shape: {latent.shape}"
    
    def test_anomaly_score(self):
        from src.models.cae import create_cae
        model = create_cae()
        model.eval()
        x = torch.randn(3, 3, 256, 256)
        scores = model.get_anomaly_score(x)
        assert scores.shape == (3,)
        assert (scores >= 0).all(), "Anomaly scores should be non-negative"
    
    def test_anomaly_map(self):
        from src.models.cae import create_cae
        model = create_cae()
        model.eval()
        x = torch.randn(2, 3, 256, 256)
        amap = model.get_anomaly_map(x)
        assert amap.shape == (2, 1, 256, 256)
    
    def test_ssim_anomaly_map(self):
        from src.models.cae import create_cae
        model = create_cae()
        model.eval()
        x = torch.randn(2, 3, 256, 256)
        amap = model.get_anomaly_map_ssim(x)
        assert amap.shape == (2, 1, 256, 256)
    
    def test_combined_anomaly_map(self):
        from src.models.cae import create_cae
        model = create_cae()
        model.eval()
        x = torch.randn(2, 3, 256, 256)
        amap = model.get_anomaly_map_combined(x, ssim_weight=0.5)
        assert amap.shape == (2, 1, 256, 256)


# =============================================================================
# VAE Tests
# =============================================================================

class TestVAE:
    """Tests for Variational Autoencoder."""
    
    def test_forward_shape(self):
        from src.models.vae import create_vae
        model = create_vae()
        x = torch.randn(2, 3, 256, 256)
        recon, mu, logvar = model(x)
        assert recon.shape == x.shape
        assert mu.shape == (2, 128)
        assert logvar.shape == (2, 128)
    
    def test_reparameterize_training(self):
        from src.models.vae import create_vae
        model = create_vae()
        model.train()
        mu = torch.randn(4, 128)
        logvar = torch.randn(4, 128)
        z = model.reparameterize(mu, logvar)
        assert z.shape == (4, 128)
    
    def test_reparameterize_eval_uses_mean(self):
        from src.models.vae import create_vae
        model = create_vae()
        model.eval()
        mu = torch.randn(4, 128)
        logvar = torch.randn(4, 128)
        z = model.reparameterize(mu, logvar)
        assert torch.allclose(z, mu), "In eval mode, reparameterize should return mu"
    
    def test_anomaly_score(self):
        from src.models.vae import create_vae
        model = create_vae()
        model.eval()
        x = torch.randn(2, 3, 256, 256)
        scores = model.get_anomaly_score(x)
        assert scores.shape == (2,)
    
    def test_kl_divergence(self):
        from src.models.vae import VariationalAutoencoder
        mu = torch.zeros(4, 128)
        logvar = torch.zeros(4, 128)
        kl = VariationalAutoencoder.kl_divergence(mu, logvar)
        assert kl.shape == (4,)
        # KL of standard normal with itself should be ~0
        assert torch.allclose(kl, torch.zeros(4), atol=1e-6)
    
    def test_loss_function(self):
        from src.models.vae import create_vae
        model = create_vae()
        x = torch.randn(2, 3, 256, 256)
        recon, mu, logvar = model(x)
        losses = model.loss_function(x, recon, mu, logvar)
        assert 'loss' in losses
        assert 'recon_loss' in losses
        assert 'kl_loss' in losses


# =============================================================================
# DAE Tests
# =============================================================================

class TestDAE:
    """Tests for Denoising Autoencoder."""
    
    def test_forward_training_adds_noise(self):
        from src.models.denoising_ae import create_denoising_ae
        model = create_denoising_ae(noise_factor=0.3)
        model.train()
        x = torch.randn(2, 3, 256, 256)
        recon, noisy = model(x)
        assert recon.shape == x.shape
        assert noisy.shape == x.shape
        assert not torch.allclose(noisy, x), "Noisy input should differ from clean"
    
    def test_forward_eval_no_noise(self):
        from src.models.denoising_ae import create_denoising_ae
        model = create_denoising_ae()
        model.eval()
        x = torch.randn(2, 3, 256, 256)
        recon, noisy = model(x)
        assert torch.allclose(noisy, x), "In eval mode, noisy should equal clean input"
    
    def test_reconstruct(self):
        from src.models.denoising_ae import create_denoising_ae
        model = create_denoising_ae()
        model.eval()
        x = torch.randn(2, 3, 256, 256)
        recon = model.reconstruct(x)
        assert recon.shape == x.shape
    
    def test_anomaly_map(self):
        from src.models.denoising_ae import create_denoising_ae
        model = create_denoising_ae()
        model.eval()
        x = torch.randn(2, 3, 256, 256)
        amap = model.get_anomaly_map(x)
        assert amap.shape == (2, 1, 256, 256)


# =============================================================================
# Skip-CAE Tests
# =============================================================================

class TestSkipCAE:
    """Tests for Skip-Connection CAE (U-Net style)."""
    
    def test_forward_shape(self):
        from src.models.skip_cae import create_skip_cae
        model = create_skip_cae()
        x = torch.randn(2, 3, 256, 256)
        out = model(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    
    def test_anomaly_score(self):
        from src.models.skip_cae import create_skip_cae
        model = create_skip_cae()
        model.eval()
        x = torch.randn(2, 3, 256, 256)
        scores = model.get_anomaly_score(x)
        assert scores.shape == (2,)
    
    def test_anomaly_map(self):
        from src.models.skip_cae import create_skip_cae
        model = create_skip_cae()
        model.eval()
        x = torch.randn(2, 3, 256, 256)
        amap = model.get_anomaly_map(x)
        assert amap.shape == (2, 1, 256, 256)
    
    def test_has_more_params_than_cae(self):
        """Skip-CAE should have more params due to skip conv layers."""
        from src.models.cae import create_cae
        from src.models.skip_cae import create_skip_cae
        cae = create_cae()
        skip = create_skip_cae()
        cae_params = sum(p.numel() for p in cae.parameters())
        skip_params = sum(p.numel() for p in skip.parameters())
        assert skip_params > cae_params


# =============================================================================
# PatchCore Tests
# =============================================================================

class TestPatchCore:
    """Tests for PatchCore feature-based detection."""
    
    def test_memory_bank_building(self):
        from src.models.patchcore import create_patchcore
        model = create_patchcore(subsample_ratio=0.5)
        normal_data = TensorDataset(torch.randn(4, 3, 256, 256))
        loader = DataLoader(normal_data, batch_size=2)
        model.fit(loader)
        assert model.memory_bank is not None
        assert model.memory_bank.shape[0] > 0
    
    def test_anomaly_score(self):
        from src.models.patchcore import create_patchcore
        model = create_patchcore(subsample_ratio=0.5)
        normal_data = TensorDataset(torch.randn(4, 3, 256, 256))
        loader = DataLoader(normal_data, batch_size=2)
        model.fit(loader)
        
        x = torch.randn(2, 3, 256, 256)
        scores = model.get_anomaly_score(x)
        assert scores.shape == (2,)
    
    def test_anomaly_map(self):
        from src.models.patchcore import create_patchcore
        model = create_patchcore(subsample_ratio=0.5)
        normal_data = TensorDataset(torch.randn(4, 3, 256, 256))
        loader = DataLoader(normal_data, batch_size=2)
        model.fit(loader)
        
        x = torch.randn(1, 3, 256, 256)
        amap = model.get_anomaly_map(x)
        assert amap.shape[0] == 1
        assert amap.shape[1] == 1
        assert amap.shape[2] == 256
        assert amap.shape[3] == 256
    
    def test_pretrained_params_frozen(self):
        from src.models.patchcore import create_patchcore
        model = create_patchcore()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable == 0, "All PatchCore backbone params should be frozen"


# =============================================================================
# CNN Classifier Tests
# =============================================================================

class TestCNN:
    """Tests for Lightweight CNN Classifier."""
    
    def test_forward_shape(self):
        from src.models.cnn_classifier import create_cnn_classifier
        model = create_cnn_classifier(num_classes=6)
        x = torch.randn(2, 3, 256, 256)
        logits = model(x)
        assert logits.shape == (2, 6)
    
    def test_predict(self):
        from src.models.cnn_classifier import create_cnn_classifier
        model = create_cnn_classifier(num_classes=6)
        model.eval()
        x = torch.randn(4, 3, 256, 256)
        preds = model.predict(x)
        assert preds.shape == (4,)
        assert (preds >= 0).all() and (preds < 6).all()
    
    def test_predict_proba(self):
        from src.models.cnn_classifier import create_cnn_classifier
        model = create_cnn_classifier(num_classes=6)
        model.eval()
        x = torch.randn(2, 3, 256, 256)
        probs = model.predict_proba(x)
        assert probs.shape == (2, 6)
        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
    
    def test_features(self):
        from src.models.cnn_classifier import create_cnn_classifier
        model = create_cnn_classifier(num_classes=6, channels=[32, 64, 128, 256])
        model.eval()
        x = torch.randn(2, 3, 256, 256)
        features = model.get_features(x)
        assert features.shape == (2, 256)
