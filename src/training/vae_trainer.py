
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from typing import Dict, Any, Optional

from src.config import DEVICE, MODELS_DIR, FIGURES_DIR
from src.data import create_mvtec_dataloaders
from src.models import create_vae
from src.training import get_optimizer, get_scheduler, EarlyStopping
from src.evaluation import plot_training_curves

def train_vae_category(
    category: str,
    config: Dict[str, Any],
    device: str = DEVICE
) -> Optional[Dict[str, Any]]:
    """
    Train VAE for a specific category with stability improvements.
    
    Args:
        category: MVTec AD category name
        config: Training configuration dictionary
        device: Torch device string
        
    Returns:
        Dictionary with results or None if failed
    """
    print(f"\n{'='*50}")
    print(f"Training VAE: {category.upper()}")
    print(f"{'='*50}")
    
    # 1. Load Data
    try:
        train_loader, test_loader = create_mvtec_dataloaders(
            category, batch_size=config['batch_size'], return_mask=True
        )
    except Exception as e:
        print(f"Skipping {category}: {e}")
        return None
    
    # 2. Create Model
    model = create_vae(latent_dim=config['latent_dim']).to(device)
    optimizer = get_optimizer(model, lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = get_scheduler(optimizer, patience=2, factor=0.5)
    early_stopping = EarlyStopping(patience=config['patience'], mode='min')
    
    # 3. Training Loop with KL Annealing
    history = {'train_loss': [], 'recon_loss': [], 'kl_loss': []}
    
    beta_max = config.get('beta_max', 0.001)
    beta_warmup_epochs = config.get('beta_warmup_epochs', 10)
    num_epochs = config['num_epochs']
    
    model.train()
    
    for epoch in tqdm(range(1, num_epochs + 1), desc=f'{category}'):
        epoch_loss, epoch_recon, epoch_kl = 0.0, 0.0, 0.0
        
        # KL annealing: linearly increase beta from 0 to beta_max
        if epoch <= beta_warmup_epochs:
            beta = beta_max * (epoch / beta_warmup_epochs)
        else:
            beta = beta_max
            
        for batch in train_loader:
            # Handle both (images, labels) and (images, masks, labels) formats
            if isinstance(batch, (list, tuple)):
                images = batch[0].to(device)
            else:
                images = batch.to(device)
                
            optimizer.zero_grad()
            
            recon, mu, logvar = model(images)
            
            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(recon, images)
            
            # KL divergence
            # -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / images.size(0)
            
            # Check for NaNs
            if torch.isnan(recon_loss) or torch.isnan(kl_loss):
                print(f"Warning: NaN loss detected at epoch {epoch}. Recon: {recon_loss}, KL: {kl_loss}")
                # Optimization step skipped for this batch
                continue
                
            # Total loss with annealed beta
            loss = recon_loss + beta * kl_loss
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
        
        n_batches = len(train_loader)
        if n_batches > 0:
            avg_loss = epoch_loss / n_batches
            avg_recon = epoch_recon / n_batches
            avg_kl = epoch_kl / n_batches
        else:
            avg_loss, avg_recon, avg_kl = 0.0, 0.0, 0.0
        
        history['train_loss'].append(avg_loss)
        history['recon_loss'].append(avg_recon)
        history['kl_loss'].append(avg_kl)
        
        # Validation or Scheduler step
        scheduler.step(avg_loss)
        
        if early_stopping(avg_loss):
            print(f"Early stopping at epoch {epoch}")
            # Ensure model is in eval mode before potential break related evaluation
            break
    
    # 4. Evaluate
    model.eval()
    all_scores, all_labels = [], []
    
    try:
        with torch.no_grad():
            for batch in test_loader:
                # Unpack batch depending on what dataloader returns
                if len(batch) == 3:
                    img, mask, label = batch
                elif len(batch) == 2:
                    img, label = batch
                else:
                    img = batch[0]
                    label = batch[1] # Guessing label index
                
                img = img.to(device)
                
                # Get anomaly score
                scores = model.get_anomaly_score(img)
                
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(label.numpy())
        
        # Calculate AUC if we have both classes
        if len(set(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_scores)
            print(f"{category.upper()} ROC-AUC: {auc:.4f}")
        else:
            print(f"{category.upper()} - Only one class in test set, skipping AUC.")
            auc = 0.5
            
    except Exception as e:
        print(f"Evaluation error for {category}: {e}")
        auc = 0.0
    
    # 5. Save model
    save_path = MODELS_DIR / f'vae_{category}_final.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history,
        'auc': auc,
    }, save_path)
    
    # 6. Plot curves
    try:
        plot_training_curves(history, title=f'VAE Training - {category}',
                             save_path=FIGURES_DIR / f'vae_{category}_training.png')
    except Exception as e:
        print(f"Plotting error: {e}")
        
    return {'category': category, 'auc': auc, 'final_loss': history['train_loss'][-1] if history['train_loss'] else 0.0}
