"""
Deep Unfolded BM3D - Utility Functions
"""

import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path


def get_device():
    """Get CUDA device if available"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def img_to_patches_and_coords(img, patch_size=16, stride=4):
    """
    Extract patches from image using unfold.

    Args:
        img: [B, C, H, W] tensor
        patch_size: int
        stride: int

    Returns:
        patches: [N, C*ps*ps]
        coords: [N, 2] (y, x coordinates)
        grid_hw: (grid_h, grid_w)
    """
    B, C, H, W = img.shape

    # Unfold: [B, C, ps, ps, grid_h, grid_w]
    patches = F.unfold(img, kernel_size=patch_size, stride=stride)
    # patches: [B, C*ps*ps, N_patches_total]

    N_patches = patches.shape[2]
    patches = patches.permute(0, 2, 1).reshape(-1, patches.shape[1])  # [B*N, C*ps*ps]

    # Compute grid coordinates
    grid_h = (H - patch_size) // stride + 1
    grid_w = (W - patch_size) // stride + 1

    coords = []
    for i in range(grid_h):
        for j in range(grid_w):
            coords.append([i * stride, j * stride])
    coords = np.array(coords)

    return patches, coords, (grid_h, grid_w)


def local_knn_for_patch(idx, patches, coords, grid_hw, stride, window, topk=16):
    """
    Find K-NN patches within local window using DC-aligned L2 distance.

    Args:
        idx: patch index
        patches: [N, D] all patches
        coords: [N, 2] patch coordinates
        grid_hw: (grid_h, grid_w)
        stride: int
        window: local window size
        topk: number of neighbors

    Returns:
        nn_idx: [topk] indices of similar patches
    """
    ref_patch = patches[idx]
    ref_coord = coords[idx]

    # DC-aligned (subtract mean)
    ref_patch_dc = ref_patch - ref_patch.mean()
    patches_dc = patches - patches.mean(axis=1, keepdims=True)

    # L2 distance
    dist = np.linalg.norm(patches_dc - ref_patch_dc, axis=1)

    # Find patches in local window
    y0, x0 = ref_coord
    window_mask = (
        (coords[:, 0] >= y0 - window // 2)
        & (coords[:, 0] <= y0 + window // 2)
        & (coords[:, 1] >= x0 - window // 2)
        & (coords[:, 1] <= x0 + window // 2)
    )

    window_indices = np.where(window_mask)[0]
    window_dist = dist[window_indices]

    # Top-K
    topk_local = min(topk, len(window_indices))
    topk_idx = window_indices[np.argsort(window_dist)[:topk_local]]

    return topk_idx


def get_kaiser2d(patch_size, beta=2.0):
    """Generate 2D Kaiser window for aggregation"""
    k1d = torch.kaiser_window(patch_size, periodic=False, beta=beta)
    k2d = torch.outer(k1d, k1d)
    return k2d


def denoise_npz_residual(model, noisy_img, config, device="cpu", tile_size=256):
    """
    Full inference pipeline: patch -> group -> predict -> aggregate

    Args:
        model: trained model
        noisy_img: [H, W] numpy array
        config: config dict
        device: torch device
        tile_size: for tiling large images

    Returns:
        denoised: [H, W] numpy array
    """
    H, W = noisy_img.shape
    patch_size = config["data"]["patch_size"]
    stride = config["data"]["stride"]
    window = config["data"]["window_size"]
    topk = config["data"]["topk"]
    kaiser_beta = config["inference"]["kaiser_beta"]

    # Convert to tensor with boundary padding (reflect mode)
    pad = patch_size // 2
    img_padded = np.pad(noisy_img, ((pad, pad), (pad, pad)), mode="reflect")
    img_tensor = torch.from_numpy(img_padded).float().to(device)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H_pad, W_pad]

    # Extract patches
    patches, coords, grid_hw = img_to_patches_and_coords(img_tensor, patch_size, stride)
    patches = patches.cpu().numpy()

    # Process patches
    denoised_patches = []
    kaiser = get_kaiser2d(patch_size, beta=kaiser_beta).to(device)

    model.eval()
    with torch.no_grad():
        for i in range(len(patches)):
            # Find similar patches
            nn_idx = local_knn_for_patch(
                i, patches, coords, grid_hw, stride, window, topk
            )

            # Group patches
            group = patches[nn_idx]  # [topk, ps*ps]
            group = group.reshape(topk, 1, patch_size, patch_size)
            group = torch.from_numpy(group).float().to(device)
            group = group.reshape(1, topk, patch_size, patch_size)

            # Predict residual
            residual = model(group)  # [1, 1, ps, ps]
            denoised_patch = group[0, 0] + residual[0, 0]
            denoised_patches.append(denoised_patch.cpu().numpy())

    # Aggregate patches
    denoised_padded = np.zeros_like(img_padded)
    weights_padded = np.zeros_like(img_padded)

    for i, patch in enumerate(denoised_patches):
        y, x = coords[i]
        y, x = int(y), int(x)

        window_weight = kaiser.cpu().numpy()
        denoised_padded[y : y + patch_size, x : x + patch_size] += patch * window_weight
        weights_padded[y : y + patch_size, x : x + patch_size] += window_weight

    # Normalize and remove padding
    denoised_padded = denoised_padded / (weights_padded + 1e-8)
    denoised = denoised_padded[pad : pad + H, pad : pad + W]

    return denoised
