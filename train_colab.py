"""
Deep Unfolded BM3D - Google Colab Training Script
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from pathlib import Path
from tqdm import tqdm

from models import get_model, count_parameters
from data import build_dataloaders
from utils import get_device, set_seed


def train_colab(config):
    """Training script optimized for Google Colab"""
    device = get_device()
    set_seed(42)
    
    print("="*60)
    print("Deep Unfolded BM3D - Training on Colab")
    print("="*60)
    
    # Model
    model = get_model(
        model_type='du_bm3d',
        in_ch=config['model']['in_channels'],
        base=config['model']['base_channels'],
        depth=config['model']['depth']
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Device: {device}")
    
    # Data
    print(f"\nLoading data from: {config['paths']['train_data']}")
    train_loader, val_loader = build_dataloaders(
        config,
        config['paths']['train_data'],
        config['paths']['val_data'],
        batch_size=config['training']['batch_size']
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    
    # Training
    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\n--- Epoch {epoch}/{config['training']['epochs']} ---")
        
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc="Train")
        for noisy_group, clean_group in pbar:
            noisy_group = noisy_group.to(device)
            clean_group = clean_group.to(device)
            
            optimizer.zero_grad()
            
            B, K, ps, _ = noisy_group.shape
            residual = model(noisy_group)
            denoised = noisy_group[:, 0:1, :, :] + residual
            
            loss = criterion(denoised, clean_group[:, 0:1, :, :])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item():.6f})
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for noisy, clean in tqdm(val_loader, desc="Val"):
                noisy = noisy.to(device)
                clean = clean.to(device)
                
                B, _, ps, _ = noisy.shape
                noisy_group = noisy.unsqueeze(1)
                residual = model(noisy_group)
                denoised = noisy + residual
                
                loss = criterion(denoised, clean)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Train Loss: {avg_train_loss:.6f}")
        print(f"Val Loss:   {avg_val_loss:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = Path(config['paths']['checkpoints_dir']) / 'best_model.pth'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Saved best model")
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    train_colab(config)
