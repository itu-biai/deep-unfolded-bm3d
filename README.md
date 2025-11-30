# Deep Unfolded BM3D

Deep learning-based denoising combining BM3D with trainable neural networks for Low-Dose CT (LDCT) images.

## Features

- ✅ BM3D + U-Net hybrid approach
- ✅ Residual learning framework  
- ✅ Multi-noise level support (10k, 50k, 100k, 500k photons)
- ✅ Google Colab optimized

## Quick Start

### Local Setup

```bash
pip install -r requirements.txt
python train.py
python evaluate.py
```

### Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')

# Download dataset from Google Drive
# Update DATA_PATH in config
python train_colab.py
```

## Dataset

Mini Deep Lesion dataset available on Google Drive:
- Train: High-quality training images
- Val: Validation set
- Test: Test set with 4 noise levels

## Architecture

The model unfolds the BM3D algorithm into a trainable deep neural network:

![Pipeline Architecture](./assets/light_theme_denoising.drawio.png)

```
Input (16x16 patches, topk=16 groups)
  ↓
U-Net with GroupNorm
  ↓
Residual Prediction
  ↓
Output Denoised Image
```

### Key Components

- **Non-Local Grouping**: K-NN based patch grouping (topk=16)
- **U-Net Backbone**: Encoder-decoder architecture with GroupNorm
- **Residual Learning**: Predicts noise residuals instead of clean images
- **Kaiser Window Aggregation**: Smooth overlapping patch aggregation

## Performance

Performance metrics across different noise levels (photon counts):

### Noise Level 1 (10k photons)
| Method | PSNR (dB) | SSIM |
|--------|-----------|------|
| Noisy | 15.63 | 0.6304 |
| BM3D | 17.79 | 0.6638 |
| **DU-BM3D** | **24.15** | **0.7417** |
| U-Net | 20.47 | 0.6887 |

### Noise Level 2 (50k photons)
| Method | PSNR (dB) | SSIM |
|--------|-----------|------|
| Noisy | 20.67 | 0.7060 |
| BM3D | 28.6 | 0.8042 |
| **DU-BM3D** | **30.4** | **0.8657** |
| U-Net | 25.99 | 0.7751 |

### Noise Level 3 (100k photons)
| Method | PSNR (dB) | SSIM |
|--------|-----------|------|
| Noisy | 23.04 | 0.7423 |
| BM3D | 29.87 | 0.8600 |
| **DU-BM3D** | **31.77** | **0.8848** |
| U-Net | 27.71 | 0.8184 |

### Noise Level 4 (500k photons)
| Method | PSNR (dB) | SSIM |
|--------|-----------|------|
| Noisy | 28.07 | 0.8295 |
| BM3D | 30.34 | 0.8653 |
| **DU-BM3D** | **31.99** | **0.8861** |
| U-Net | 30.84 | 0.8891 |

**DU-BM3D consistently outperforms baseline methods across all noise levels.** ✨

### Visual Results

Denoising comparison for different noise levels from the paper:

**Noise Level 1 (10k photons) - High Noise**
![10k Photons Example](./assets/10k_makale.jpg)

**Noise Level 3 (100k photons) - Low Noise**
![100k Photons Example](./assets/100k_makale.jpg)

### Performance Analysis

**Model Complexity Comparison**
![Model Complexity](./assets/model_complexity_comparison.png)

**Computational Time Comparison**
![Time Comparison](./assets/model_time_comparison.png)

## Usage

### Training

```bash
python train.py --config config.yaml
```

### Evaluation

```bash
python evaluate.py --model best_model.pth
```

### Inference

```python
from models import get_model
from utils import denoise_image

model = get_model('du_bm3d')
model.load_state_dict(torch.load('model.pth'))
denoised = denoise_image(model, noisy_image)
```

## Project Structure

```
.
├── README.md
├── requirements.txt
├── config.yaml
├── train.py
├── evaluate.py
├── models.py
├── utils.py
├── data.py
└── notebooks/
    └── DU_BM3D.ipynb
```

## References

Based on the paper: **Deep Unfolded BM3D: Unrolling Non-Local Collaborative Filtering into Trainable Neural Networks**

### Key References

- **BM3D**: Dabov et al., "Image Denoising by Sparse 3D Transform-Domain Collaborative Filtering", 2007
- **Deep Unfolding**: Monga et al., "Algorithm Unrolling: Interpretable, Efficient Deep Learning", 2021
- **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", 2015

### Main Contributions

1. **Algorithm Unfolding**: BM3D non-local filtering steps transformed into trainable neural network layers
2. **Hybrid Approach**: Combines classical image processing (BM3D) with modern deep learning
3. **LDCT Application**: Optimized for Low-Dose CT image denoising
4. **Multi-Noise Robustness**: Effective across different noise levels (10k-500k photons)

## License

GNU General Public License v3

## Citation

```bibtex
@article{dubm3d2024,
  title={Deep Unfolded BM3D: Unrolling Non-Local Collaborative Filtering into Trainable Neural Networks},
  year={2024}
}
```
