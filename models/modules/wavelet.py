"""
Wavelet Feature Enhancement Module for Weld-YOLO11-SO
Specialized for Small Welding Defect Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def haar_dwt2d(x):
    """
    Differentiable 2D Discrete Wavelet Transform (Haar Wavelet) using PyTorch tensor operations.
    Handles both even and odd spatial dimensions safely via padding.
    Splits input [B, C, H, W] into 4 sub-bands of shape [B, C, ceil(H/2), ceil(W/2)]:
      - LL: Low-frequency approximation
      - LH: Horizontal detail
      - HL: Vertical detail
      - HH: Diagonal high-frequency detail
    """
    B, C, H, W = x.shape
    pad_h = H % 2
    pad_w = W % 2
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    LL = (x00 + x01 + x10 + x11) * 0.5
    LH = (-x00 - x01 + x10 + x11) * 0.5
    HL = (-x00 + x01 - x10 + x11) * 0.5
    HH = (x00 - x01 - x10 + x11) * 0.5

    return LL, LH, HL, HH, (H, W)


def haar_idwt2d(LL, LH, HL, HH, orig_shape=None):
    """
    Differentiable 2D Inverse Discrete Wavelet Transform (Haar Wavelet).
    Reconstructs original spatial resolution [B, C, H, W] from sub-bands.
    """
    x00 = (LL - LH - HL + HH) * 0.5
    x01 = (LL - LH + HL - HH) * 0.5
    x10 = (LL + LH - HL - HH) * 0.5
    x11 = (LL + LH + HL + HH) * 0.5

    B, C, H2, W2 = LL.shape
    H, W = H2 * 2, W2 * 2

    # Interleave sub-grid outputs into full spatial grid
    out = torch.empty((B, C, H, W), device=LL.device, dtype=LL.dtype)
    out[:, :, 0::2, 0::2] = x00
    out[:, :, 0::2, 1::2] = x01
    out[:, :, 1::2, 0::2] = x10
    out[:, :, 1::2, 1::2] = x11

    if orig_shape is not None:
        orig_H, orig_W = orig_shape
        if H != orig_H or W != orig_W:
            out = out[:, :, :orig_H, :orig_W]

    return out


class WaveletBlock(nn.Module):
    """
    WaveletBlock for high-frequency welding defect texture preservation.
    
    Args:
        c1 (int): Input channel dimension.
        c2 (int, optional): Output channel dimension. Defaults to c1.
    """
    def __init__(self, c1, c2=None):
        super().__init__()
        c2 = c2 if c2 is not None else c1
        self.c1 = c1
        self.c2 = c2

        # High-frequency detail processing sub-network (LH, HL, HH)
        self.high_freq_conv = nn.Sequential(
            nn.Conv2d(c1 * 3, c1 * 3, kernel_size=3, padding=1, groups=c1, bias=False),
            nn.BatchNorm2d(c1 * 3),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1 * 3, c1 * 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1 * 3),
            nn.SiLU(inplace=True)
        )

        # Lightweight Frequency/Channel Attention Module
        self.freq_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1 * 3, max(c1 // 4, 16), kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(c1 // 4, 16), c1 * 3, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Channel projection layer if c2 != c1
        self.proj = nn.Conv2d(c1, c2, kernel_size=1, bias=False) if c2 != c1 else nn.Identity()

    def _check_and_reinit(self, x):
        in_c = x.shape[1]
        if in_c != self.c1:
            self.c1 = in_c
            device, dtype = x.device, x.dtype
            self.high_freq_conv = nn.Sequential(
                nn.Conv2d(in_c * 3, in_c * 3, kernel_size=3, padding=1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c * 3),
                nn.SiLU(inplace=True),
                nn.Conv2d(in_c * 3, in_c * 3, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_c * 3),
                nn.SiLU(inplace=True)
            ).to(device=device, dtype=dtype)

            self.freq_attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_c * 3, max(in_c // 4, 16), kernel_size=1, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(max(in_c // 4, 16), in_c * 3, kernel_size=1, bias=False),
                nn.Sigmoid()
            ).to(device=device, dtype=dtype)

            if self.c2 != self.c1 and self.c2 is not None:
                self.proj = nn.Conv2d(self.c1, self.c2, kernel_size=1, bias=False).to(device=device, dtype=dtype)
            else:
                self.proj = nn.Identity()

    def forward(self, x):
        # Auto-adapt channels if mismatch occurs due to YOLO width scaling
        self._check_and_reinit(x)

        # 1. 2D Haar Discrete Wavelet Transform
        LL, LH, HL, HH, orig_shape = haar_dwt2d(x)

        # 2. Process High-Frequency Sub-Bands (fine welding defects, edges, pinholes)
        high_freq = torch.cat([LH, HL, HH], dim=1) # [B, 3*C, H/2, W/2]
        processed_hf = self.high_freq_conv(high_freq)
        attn_weights = self.freq_attn(processed_hf)
        enhanced_hf = processed_hf * attn_weights

        # 3. Unpack Enhanced High-Frequency Components
        LH_proc, HL_proc, HH_proc = torch.chunk(enhanced_hf, 3, dim=1)

        # 4. Inverse DWT Feature Reconstruction back to original spatial shape [B, C, H, W]
        recon = haar_idwt2d(LL, LH_proc, HL_proc, HH_proc, orig_shape=orig_shape)

        # 5. Residual Connection & Channel Projection
        out = self.proj(x + recon)
        return out


if __name__ == '__main__':
    # Simple Unit Test (test both even and odd shapes)
    print("Testing WaveletBlock...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for shape in [(2, 128, 80, 80), (2, 128, 33, 33)]:
        x = torch.randn(*shape, device=device)
        block = WaveletBlock(c1=128, c2=128).to(device)
        out = block(x)
        print(f"Input shape: {x.shape} -> Output shape: {out.shape}")
        assert out.shape == shape, f"Shape mismatch: expected {shape}, got {out.shape}"
        out.sum().backward()
    print("WaveletBlock Unit Test PASSED!")
