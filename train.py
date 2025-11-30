"""
Deep Unfolded BM3D - Training Script
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


def train_epoch(model, train_loader, optimizer, criterion, device, use_amp=False, scaler=None):
    """Train one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc="Train")
    for noisy_group, clean_group in pbar:
        noisy_group = noisy_group.to(device)  # [B, topk, ps, ps]
        clean_group = clean_group.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and scaler:
            with torch.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                # Reshape for model
                B, K, ps, ps = noisy_group.shape
                noisy_group = noisy_group.reshape(B, K, ps, ps)
                
                # Model predicts residual
                residual = model(noisy_group)  # [B, 1, ps, ps]
                denoised = noisy_group[:, 0:1, :, :] + residual  # Add to first patch
                
                loss = criterion(denoised, clean_group[:, 0:1, :, :])
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            B, K, ps, ps = noisy_group.shape
            noisy_group = noisy_group.reshape(B, K, ps, ps)
            
            residual = model(noisy_group)
            denoised = noisy_group[:, 0:1, :, :] + residual
            
            loss = criterion(denoised, clean_group[:, 0:1, :, :])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item():.6f})
    
    return total_loss / len(train_loader)


def evaluate(model, val_loader, criterion, device, use_amp=False):
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for noisy, clean in tqdm(val_loader, desc="Val"):
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            if use_amp:
                with torch.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                    # For validation, treat as single patch in group
                    B, _, ps, ps = noisy.shape
                    noisy_group = noisy.unsqueeze(1)  # [B, 1, ps, ps]
                    residual = model(noisy_group)
                    denoised = noisy + residual
                    loss = criterion(denoised, clean)
            else:
                B, _, ps, ps = noisy.shape
                noisy_group = noisy.unsqueeze(1)
                residual = model(noisy_group)
                denoised = noisy + residual
                loss = criterion(denoised, clean)
            
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def train(config, model_path=None):
    """Main training loop"""
    device = get_device()
    set_seed(42)
    
    # Create directories
    Path(config['paths']['checkpoints_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['logs_dir']).mkdir(parents=True, exist_ok=True)
    
    # Model
    model = get_model(
        model_type='du_bm3d',
        in_ch=config['model']['in_channels'],
        base=config['model']['base_channels'],
        depth=config['model']['depth']
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Data
    train_loader, val_loader = build_dataloaders(
        config,
        config['paths']['train_data'],
        config['paths']['val_data'],
        batch_size=config['training']['batch_size']
    )
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['training']['learning_rate'],
        total_steps=config['training']['epochs'] * len(train_loader),
        pct_start=0.1
    )
    
    # Loss & AMP
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler() if config['training']['use_amp'] else None
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\n--- Epoch {epoch}/{config['training']['epochs']} ---")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, 
                               device, config['training']['use_amp'], scaler)
        val_loss = evaluate(model, val_loader, criterion, device, config['training']['use_amp'])
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss:   {val_loss:.6f}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = Path(config['paths']['checkpoints_dir']) / 'best_model.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Saved best model: {checkpoint_path}")
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    train(config)
