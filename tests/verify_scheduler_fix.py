"""
Verify scheduler fix for unsupervised training (no validation loader).
"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, 'F:/Thesis')
from src.training.trainer import AutoencoderTrainer, get_scheduler

def test_scheduler_no_val():
    print("Testing scheduler without validation loader...")
    
    # Dummy data and model
    model = nn.Linear(10, 10)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    
    # Create ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)
    
    trainer = AutoencoderTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device='cpu'
    )
    
    # Dummy loader
    train_data = TensorDataset(torch.randn(10, 10))
    train_loader = DataLoader(train_data, batch_size=2)
    
    # Run fit without val_loader
    try:
        trainer.fit(
            train_loader=train_loader,
            val_loader=None,
            num_epochs=2,
            loss_fn=nn.MSELoss(),
            verbose=False
        )
        print("✅ Fit completed without error!")
        
        # Verify LR wasn't reduced yet (patience=1)
        curr_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {curr_lr}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        raise

if __name__ == "__main__":
    test_scheduler_no_val()
