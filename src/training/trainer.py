"""
Training utilities for anomaly detection models.

Includes:
- Trainer class for autoencoders
- Loss functions
- Learning rate schedulers
- Training callbacks
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Optional, Callable, List
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import MODELS_DIR, LOGS_DIR, DEVICE


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class TrainingHistory:
    """Track training metrics across epochs."""
    
    def __init__(self):
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.learning_rates: List[float] = []
        self.epochs: List[int] = []
        self.extra_metrics: Dict[str, List[float]] = {}
    
    def add_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        lr: Optional[float] = None,
        **kwargs
    ):
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        if val_loss is not None:
            self.val_loss.append(val_loss)
        if lr is not None:
            self.learning_rates.append(lr)
        
        for key, value in kwargs.items():
            if key not in self.extra_metrics:
                self.extra_metrics[key] = []
            self.extra_metrics[key].append(value)
    
    def to_dict(self) -> Dict:
        return {
            'epochs': self.epochs,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'learning_rates': self.learning_rates,
            **self.extra_metrics,
        }
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingHistory':
        with open(path, 'r') as f:
            data = json.load(f)
        
        history = cls()
        history.epochs = data.get('epochs', [])
        history.train_loss = data.get('train_loss', [])
        history.val_loss = data.get('val_loss', [])
        history.learning_rates = data.get('learning_rates', [])
        
        for key, value in data.items():
            if key not in ['epochs', 'train_loss', 'val_loss', 'learning_rates']:
                history.extra_metrics[key] = value
        
        return history


class AutoencoderTrainer:
    """
    Trainer class for autoencoder models.
    
    Handles training loop, validation, checkpointing, and logging.
    
    Args:
        model: Autoencoder model (CAE, VAE, or Denoising AE)
        optimizer: PyTorch optimizer
        scheduler: Optional learning rate scheduler
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory to save training logs
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = DEVICE,
        checkpoint_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        self.checkpoint_dir = checkpoint_dir or MODELS_DIR
        self.log_dir = log_dir or LOGS_DIR
        
        self.history = TrainingHistory()
        self.best_val_loss = float('inf')
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        loss_fn: Callable,
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                images = batch[0].to(self.device)
            else:
                images = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss (handle different model types)
            if isinstance(outputs, tuple):
                # VAE returns (recon, mu, logvar)
                loss = loss_fn(images, *outputs)
            else:
                loss = loss_fn(images, outputs)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        loss_fn: Callable,
    ) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        
        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                images = batch[0].to(self.device)
            else:
                images = batch.to(self.device)
            
            outputs = self.model(images)
            
            if isinstance(outputs, tuple):
                loss = loss_fn(images, *outputs)
            else:
                loss = loss_fn(images, outputs)
            
            total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        loss_fn: Optional[Callable] = None,
        early_stopping: Optional[EarlyStopping] = None,
        save_every: int = 10,
        experiment_name: str = 'experiment',
        verbose: bool = True,
    ) -> TrainingHistory:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of training epochs
            loss_fn: Loss function (default: MSE)
            early_stopping: Optional early stopping callback
            save_every: Save checkpoint every N epochs
            experiment_name: Name for saving checkpoints
            verbose: Whether to show progress bar
            
        Returns:
            TrainingHistory with metrics
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        
        start_time = time.time()
        
        iterator = range(1, num_epochs + 1)
        if verbose:
            iterator = tqdm(iterator, desc='Training')
        
        for epoch in iterator:
            # Train
            train_loss = self.train_epoch(train_loader, loss_fn)
            
            # Validate
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader, loss_fn)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history.add_epoch(epoch, train_loss, val_loss, current_lr)
            
            # Update progress bar
            if verbose:
                desc = f'Epoch {epoch} | Train: {train_loss:.6f}'
                if val_loss is not None:
                    desc += f' | Val: {val_loss:.6f}'
                iterator.set_description(desc)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    metric = val_loss if val_loss is not None else train_loss
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()
            
            # Save best model
            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(
                    f'{experiment_name}_best.pth',
                    epoch=epoch,
                    val_loss=val_loss,
                )
            
            # Periodic checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(
                    f'{experiment_name}_epoch{epoch}.pth',
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                )
            
            # Early stopping
            if early_stopping is not None:
                monitor = val_loss if val_loss is not None else train_loss
                if early_stopping(monitor):
                    print(f'\nEarly stopping at epoch {epoch}')
                    break
        
        # Save final model
        self.save_checkpoint(
            f'{experiment_name}_final.pth',
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
        )
        
        # Save training history
        history_path = self.log_dir / f'{experiment_name}_history.json'
        self.history.save(str(history_path))
        
        elapsed = time.time() - start_time
        print(f'\nTraining completed in {elapsed/60:.2f} minutes')
        
        return self.history
    
    def save_checkpoint(self, filename: str, **kwargs):
        """Save model checkpoint."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'timestamp': datetime.now().isoformat(),
            **kwargs,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint


def get_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    optimizer_type: str = 'adam',
) -> optim.Optimizer:
    """
    Create optimizer for model.
    
    Args:
        model: PyTorch model
        lr: Learning rate
        weight_decay: L2 regularization
        optimizer_type: 'adam', 'adamw', or 'sgd'
        
    Returns:
        Optimizer instance
    """
    if optimizer_type == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def get_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = 'plateau',
    **kwargs,
) -> optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: 'plateau', 'step', 'cosine', or 'exponential'
        **kwargs: Additional scheduler arguments
        
    Returns:
        Scheduler instance
    """
    if scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 5),
        )
    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1),
        )
    elif scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 100),
        )
    elif scheduler_type == 'exponential':
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.95),
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
