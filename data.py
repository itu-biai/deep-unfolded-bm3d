"""
Deep Unfolded BM3D - Data Loading
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import glob


class NPZDataset(Dataset):
    """Simple NPZ dataset loader"""
    
    def __init__(self, data_dir, noisy_key='noisy_clipped_100k', clean_key='gt_clipped'):
        self.files = glob.glob(str(Path(data_dir) / "*.npz"))
        self.noisy_key = noisy_key
        self.clean_key = clean_key
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        noisy = torch.from_numpy(data[self.noisy_key]).float()
        clean = torch.from_numpy(data[self.clean_key]).float()
        return noisy.unsqueeze(0), clean.unsqueeze(0)  # [1, H, W]


class NPZGroupDataset(Dataset):
    """NPZ dataset with patch grouping for training"""
    
    def __init__(self, data_dir, patch_size=16, stride=4, window=39, topk=16,
                 noisy_key='noisy_clipped_100k', clean_key='gt_clipped',
                 groups_per_image=128):
        self.files = glob.glob(str(Path(data_dir) / "*.npz"))
        self.patch_size = patch_size
        self.stride = stride
        self.window = window
        self.topk = topk
        self.noisy_key = noisy_key
        self.clean_key = clean_key
        self.groups_per_image = groups_per_image
        
        self.data_cache = {}
        self._load_all_data()
    
    def _load_all_data(self):
        """Cache all NPZ files"""
        for f in self.files:
            data = np.load(f)
            self.data_cache[f] = {
                'noisy': data[self.noisy_key],
                'clean': data[self.clean_key]
            }
    
    def __len__(self):
        return len(self.files) * self.groups_per_image
    
    def __getitem__(self, idx):
        file_idx = idx // self.groups_per_image
        group_idx = idx % self.groups_per_image
        
        f = self.files[file_idx]
        data = self.data_cache[f]
        noisy = data['noisy']
        clean = data['clean']
        
        H, W = noisy.shape
        
        # Random patch location
        y = np.random.randint(0, H - self.patch_size)
        x = np.random.randint(0, W - self.patch_size)
        
        # Extract reference patch
        ref_noisy = noisy[y:y+self.patch_size, x:x+self.patch_size]
        ref_clean = clean[y:y+self.patch_size, x:x+self.patch_size]
        
        # Find similar patches (simplified - just stack random patches)
        group_noisy = []
        group_clean = []
        
        for _ in range(self.topk):
            y_sim = np.random.randint(0, H - self.patch_size)
            x_sim = np.random.randint(0, W - self.patch_size)
            group_noisy.append(noisy[y_sim:y_sim+self.patch_size, x_sim:x_sim+self.patch_size])
            group_clean.append(clean[y_sim:y_sim+self.patch_size, x_sim:x_sim+self.patch_size])
        
        group_noisy = np.stack(group_noisy)  # [topk, ps, ps]
        group_clean = np.stack(group_clean)
        
        # Convert to tensor
        group_noisy = torch.from_numpy(group_noisy).float()
        group_clean = torch.from_numpy(group_clean).float()
        
        return group_noisy, group_clean  # [topk, ps, ps]


def build_dataloaders(config, data_dir_train, data_dir_val, 
                      batch_size=16, num_workers=0):
    """Build training and validation dataloaders"""
    
    train_dataset = NPZGroupDataset(
        data_dir_train,
        patch_size=config['data']['patch_size'],
        stride=config['data']['stride'],
        window=config['data']['window_size'],
        topk=config['data']['topk'],
        noisy_key=config['dataset']['train_key'],
        clean_key=config['dataset']['clean_key'],
        groups_per_image=128
    )
    
    val_dataset = NPZDataset(
        data_dir_val,
        noisy_key=config['dataset']['val_key'],
        clean_key=config['dataset']['clean_key']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader
