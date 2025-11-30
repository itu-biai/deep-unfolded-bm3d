"""
Deep Unfolded BM3D - Evaluation Script
"""
import argparse
import torch
import numpy as np
import yaml
from pathlib import Path
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import glob
from tqdm import tqdm

from models import get_model
from data import NPZDataset
from utils import get_device, denoise_npz_residual


def evaluate_method(method_name, model, test_loader, device, config):
    """Evaluate single method"""
    print(f"\n{'='*60}")
    print(f"Evaluating {method_name}")
    print(f"{'='*60}")
    
    psnr_list = []
    ssim_list = []
    
    with torch.no_grad():
        for noisy, clean in tqdm(test_loader):
            noisy_np = noisy.squeeze().numpy()
            clean_np = clean.squeeze().numpy()
            
            if method_name == "DU-BM3D":
                # Full pipeline
                denoised = denoise_npz_residual(model, noisy_np, config, device)
            else:
                denoised = noisy_np
            
            # Compute metrics
            psnr_val = psnr(clean_np, denoised, data_range=clean_np.max()-clean_np.min())
            ssim_val = ssim(clean_np, denoised, data_range=clean_np.max()-clean_np.min())
            
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
    
    mean_psnr = np.mean(psnr_list)
    mean_ssim = np.mean(ssim_list)
    
    print(f"{method_name:15s}: PSNR = {mean_psnr:6.2f} dB, SSIM = {mean_ssim:6.4f}")
    
    return mean_psnr, mean_ssim


def evaluate(config, model_path):
    """Evaluate model on test set"""
    device = get_device()
    
    # Load model
    model = get_model(
        model_type='du_bm3d',
        in_ch=config['model']['in_channels'],
        base=config['model']['base_channels'],
        depth=config['model']['depth']
    ).to(device)
    
    if model_path:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Loaded model: {model_path}")
    else:
        raise ValueError("Model path is required for evaluation")
    
    model.eval()
    
    # Test dataset
    test_data_path = Path(config['paths']['test_data'])
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data path not found: {test_data_path}")
    
    test_loader = torch.utils.data.DataLoader(
        NPZDataset(
            config['paths']['test_data'],
            noisy_key=config['dataset']['test_keys'][2],  # 100k
            clean_key=config['dataset']['clean_key']
        ),
        batch_size=1,
        shuffle=False
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    evaluate_method("DU-BM3D", model, test_loader, device, config)
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--model', type=str, default='outputs/checkpoints/best_model.pth', 
                       help='Model checkpoint')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    evaluate(config, args.model)
