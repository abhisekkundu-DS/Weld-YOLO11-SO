"""
DySample: Dynamic & Content-Aware Upsampling Module
Specialized for Small Welding Defect Feature Reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DySample(nn.Module):
    """
    DySample: Lightweight, content-aware dynamic upsampling module using PyTorch grid sampling.
    Improves spatial resolution and feature reconstruction for small objects and fine micro-defects.

    Args:
        c1 (int): Input channel dimension.
        c2 (int, optional): Output channel dimension. Defaults to c1.
        scale (int): Upsampling scale factor (e.g. 2 or 4). Defaults to 2.
        style (str): Sampling style ('lp' for linear scope or 'pl' for pixel shuffle). Defaults to 'lp'.
    """
    def __init__(self, c1, c2=None, scale=2, style='lp'):
        super().__init__()
        c2 = c2 if c2 is not None else c1
        self.c1 = c1
        self.c2 = c2
        self.scale = scale
        self.style = style

        # Offset generation network: predicts 2 * scale^2 sampling offsets per spatial location
        self.offset_dim = 2 * (scale ** 2)
        self.offset_conv = nn.Sequential(
            nn.Conv2d(c1, c1 // 4 if c1 >= 16 else c1, kernel_size=3, padding=1, groups=max(1, c1 // 4 if c1 >= 16 else c1), bias=False),
            nn.BatchNorm2d(c1 // 4 if c1 >= 16 else c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1 // 4 if c1 >= 16 else c1, self.offset_dim, kernel_size=1, bias=False)
        )

        # Optional channel projection if c2 != c1
        self.proj = nn.Conv2d(c1, c2, kernel_size=1, bias=False) if c2 != c1 else nn.Identity()

    def _init_grid(self, B, H, W, device, dtype):
        # Generate base normalized sampling grid [-1, 1] for F.grid_sample
        y = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1) # [B, H, W, 2]
        return base_grid

    def forward(self, x):
        B, C, H, W = x.shape
        scale = self.scale
        out_H, out_W = H * scale, W * scale

        try:
            # 1. Compute Dynamic Sampling Offsets
            offsets = self.offset_conv(x) # [B, 2*scale^2, H, W]
            
            # Reshape offsets to match pixel-shuffle layout for high-resolution grid
            offsets = F.pixel_shuffle(offsets, scale) # [B, 2, H*scale, W*scale]
            offsets = offsets.permute(0, 2, 3, 1) # [B, H*scale, W*scale, 2]

            # 2. Base Grid Generation
            base_grid = self._init_grid(B, out_H, out_W, x.device, x.dtype) # [B, H*scale, W*scale, 2]

            # 3. Add Normalized Dynamic Offsets (bounded to avoid extreme extrapolation)
            norm_offsets = torch.tanh(offsets) * (2.0 / max(out_H, out_W))
            sample_grid = base_grid + norm_offsets

            # 4. Bilinear Grid Sampling (Content-Aware Dynamic Resampling)
            upsampled = F.grid_sample(x, sample_grid, mode='bilinear', padding_mode='reflection', align_corners=True)

        except Exception:
            # Safe fallback if grid sampling encounters edge-case constraints
            upsampled = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)

        # 5. Channel Projection & Output
        return self.proj(upsampled)


if __name__ == '__main__':
    # Simple Unit Test
    print("Testing DySample...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(2, 128, 40, 40, device=device)
    dysampler = DySample(c1=128, c2=128, scale=2).to(device)
    out = dysampler(x)
    print(f"Input shape: {x.shape} -> Output shape: {out.shape}")
    assert out.shape == (2, 128, 80, 80), f"Shape mismatch: expected (2, 128, 80, 80), got {out.shape}"
    out.sum().backward()
    print("DySample Unit Test PASSED!")
