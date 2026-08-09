"""
WeldSimAM: Lightweight Directional & Structural Attention Module
Specialized for Industrial Welding Defect Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeldSimAM(nn.Module):
    """
    WeldSimAM: Lightweight 3D SimAM attention combined with Directional Feature Extraction.
    Designed for small welding defect detection (cracks, porosity, lack of fusion).
    
    Args:
        c1 (int): Input channel dimension.
        c2 (int, optional): Output channel dimension. Defaults to c1.
        e_lambda (float): Hyperparameter for SimAM energy calculation. Defaults to 1e-4.
    """
    def __init__(self, c1, c2=None, e_lambda=1e-4):
        super().__init__()
        c2 = c2 if c2 is not None else c1
        self.c1 = c1
        self.c2 = c2
        self.e_lambda = e_lambda

        # Directional feature extraction pooling
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1)) # [B, C, H, 1]
        self.pool_w = nn.AdaptiveAvgPool2d((1, None)) # [B, C, 1, W]

        # Directional feature transformation layers
        mip = max(8, c1 // 16)
        self.conv1 = nn.Conv2d(c1, mip, kernel_size=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=min(mip, 8), num_channels=mip)
        self.act1 = nn.SiLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, c1, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mip, c1, kernel_size=1, bias=False)

        # Output projection if c2 != c1
        self.proj = nn.Conv2d(c1, c2, kernel_size=1, bias=False) if c2 != c1 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        # --- 1. SimAM Parameter-Free 3D Energy Attention ---
        # Spatial size n = H * W - 1
        n = H * W - 1
        # Mean & Variance calculation across spatial dimensions per channel
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        
        # Energy formula e_t and 3D attention weight M
        # e_t = 4 * (v + e_lambda) / (d + 2 * v + 2 * e_lambda)
        # 1 / e_t = (d + 2*v + 2*e_lambda) / (4 * (v + e_lambda))
        simam_weight = torch.sigmoid((d / (4 * (v + self.e_lambda))) + 0.5)

        # --- 2. Directional Feature Extraction (Horizontal & Vertical) ---
        x_h = self.pool_h(x) # [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2) # [B, C, W, 1]

        # Concat along spatial dimension for joint transformation
        y = torch.cat([x_h, x_w], dim=2) # [B, C, H + W, 1]
        y = self.act1(self.gn1(self.conv1(y)))

        # Split back to horizontal and vertical pathways
        x_h_feat, x_w_feat = torch.split(y, [H, W], dim=2)
        x_w_feat = x_w_feat.permute(0, 1, 3, 2) # [B, C, 1, W]

        # Directional attention maps
        a_h = torch.sigmoid(self.conv_h(x_h_feat)) # [B, C, H, 1]
        a_w = torch.sigmoid(self.conv_w(x_w_feat)) # [B, C, 1, W]

        dir_weight = a_h * a_w # [B, C, H, W]

        # --- 3. Attention Fusion & Reweighting ---
        out = x * simam_weight * dir_weight

        # --- 4. Channel Projection & Output ---
        return self.proj(out)


if __name__ == '__main__':
    # Simple Unit Test
    print("Testing WeldSimAM...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(2, 128, 80, 80, device=device)
    simam = WeldSimAM(c1=128, c2=128).to(device)
    out = simam(x)
    print(f"Input shape: {x.shape} -> Output shape: {out.shape}")
    assert out.shape == (2, 128, 80, 80), f"Shape mismatch: expected (2, 128, 80, 80), got {out.shape}"
    out.sum().backward()
    print("WeldSimAM Unit Test PASSED!")
