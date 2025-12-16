"""
Deep Unfolded BM3D - Model Architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double Convolution Block with GroupNorm"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DUBM3D(nn.Module):
    """Deep Unfolded BM3D - Residual Learning U-Net"""

    def __init__(self, in_ch=16, base=32, depth=2, out_ch=1):
        super().__init__()

        self.in_conv = DoubleConv(in_ch, base)

        # Encoder
        self.down_blocks = nn.ModuleList()
        channels = [base]
        for i in range(depth):
            prev_ch = channels[-1]
            next_ch = prev_ch * 2
            self.down_blocks.append(
                nn.Sequential(nn.MaxPool2d(2), DoubleConv(prev_ch, next_ch))
            )
            channels.append(next_ch)

        # Decoder
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            in_ch_up = channels[-(i + 1)]
            skip_ch = channels[-(i + 2)]
            out_ch_up = in_ch_up // 2
            self.up_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(in_ch_up, out_ch_up, 1),
                    DoubleConv(out_ch_up + skip_ch, out_ch_up),
                )
            )

        self.out_conv = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x):
        # Input: [B, in_ch, H, W]
        x = self.in_conv(x)

        # Encoder
        skips = [x]
        for block in self.down_blocks:
            x = block(x)
            skips.append(x)

        # Decoder
        for i, block in enumerate(self.up_blocks):
            x = block[0](x)  # Upsample
            x = block[1](x)  # 1x1 conv
            skip = skips[-(i + 2)]
            x = torch.cat([x, skip], dim=1)
            x = block[2](x)  # DoubleConv

        # Output residual
        residual = self.out_conv(x)
        return residual


class UNet2D(nn.Module):
    """Classic 2D U-Net for comparison"""

    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()

        self.enc1 = DoubleConv(in_ch, 64)
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.enc4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        self.dec4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(512, 256, 1),
            DoubleConv(512, 256),
        )
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 1),
            DoubleConv(256, 128),
        )
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 1),
            DoubleConv(128, 64),
        )

        self.out = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d4 = self.dec4[0](e4)
        d4 = self.dec4[1](d4)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4[2](d4)

        d3 = self.dec3[0](d4)
        d3 = self.dec3[1](d3)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3[2](d3)

        d2 = self.dec2[0](d3)
        d2 = self.dec2[1](d2)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2[2](d2)

        return self.out(d2)


def get_model(model_type="du_bm3d", in_ch=16, base=32, depth=2):
    """Factory function to get model"""
    if model_type == "du_bm3d":
        return DUBM3D(in_ch=in_ch, base=base, depth=depth)
    elif model_type == "unet":
        return UNet2D(in_ch=1, out_ch=1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
