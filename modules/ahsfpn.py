"""
AHSFPN: Adaptive Hybrid Spatial Feature Pyramid Network
Specialized for Multi-Scale Welding Defect Feature Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv-BN-SiLU Building Block for AHSFPN Feature Fusion."""
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class AHSFPN(nn.Module):
    """
    AHSFPN: Adaptive Hybrid Spatial Feature Pyramid Network for multi-scale welding defect detection.
    Combines P2, P3, P4, P5 feature maps using Top-Down and Bottom-Up pathways with learnable adaptive weights.

    Args:
        in_channels (list of int): Channel dimensions for input feature maps [P2, P3, P4, P5].
        out_channels (list or int): Output channel dimensions for [P2_out, P3_out, P4_out, P5_out].
    """
    def __init__(self, in_channels=[128, 256, 512, 1024], out_channels=[128, 256, 512, 1024]):
        super().__init__()
        if isinstance(in_channels, int):
            in_channels = [in_channels] * 4
        if isinstance(out_channels, int):
            out_channels = [out_channels] * 4

        c2_in, c3_in, c4_in, c5_in = in_channels
        c2_out, c3_out, c4_out, c5_out = out_channels

        # 1x1 Conv Lateral Projection Layers to standardize channel dimensions
        self.proj_p2 = nn.Conv2d(c2_in, c2_out, kernel_size=1, bias=False)
        self.proj_p3 = nn.Conv2d(c3_in, c3_out, kernel_size=1, bias=False)
        self.proj_p4 = nn.Conv2d(c4_in, c4_out, kernel_size=1, bias=False)
        self.proj_p5 = nn.Conv2d(c5_in, c5_out, kernel_size=1, bias=False)

        # Upsampling adapter Convs for top-down pathway
        self.up_p5_to_p4 = nn.Conv2d(c5_out, c4_out, kernel_size=1, bias=False)
        self.up_p4_to_p3 = nn.Conv2d(c4_out, c3_out, kernel_size=1, bias=False)
        self.up_p3_to_p2 = nn.Conv2d(c3_out, c2_out, kernel_size=1, bias=False)

        # Downsampling adapter Convs for bottom-up pathway
        self.down_p2_to_p3 = ConvBlock(c2_out, c3_out, k=3, s=2, p=1)
        self.down_p3_to_p4 = ConvBlock(c3_out, c4_out, k=3, s=2, p=1)
        self.down_p4_to_p5 = ConvBlock(c4_out, c5_out, k=3, s=2, p=1)

        # Smooth Convs after feature fusion
        self.smooth_td_p4 = ConvBlock(c4_out, c4_out, k=3, s=1, p=1)
        self.smooth_td_p3 = ConvBlock(c3_out, c3_out, k=3, s=1, p=1)
        self.smooth_td_p2 = ConvBlock(c2_out, c2_out, k=3, s=1, p=1)

        self.smooth_bu_p3 = ConvBlock(c3_out, c3_out, k=3, s=1, p=1)
        self.smooth_bu_p4 = ConvBlock(c4_out, c4_out, k=3, s=1, p=1)
        self.smooth_bu_p5 = ConvBlock(c5_out, c5_out, k=3, s=1, p=1)

        # Learnable Adaptive Fusion Weights (Softmax normalized during forward)
        self.w_td_p4 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w_td_p3 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w_td_p2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

        self.w_bu_p3 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w_bu_p4 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w_bu_p5 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

    def forward(self, features):
        """
        Args:
            features (list of Tensors): [P2, P3, P4, P5]
              - P2: [B, C2, 256, 256] (Stride 4)
              - P3: [B, C3, 128, 128] (Stride 8)
              - P4: [B, C4, 64, 64]   (Stride 16)
              - P5: [B, C5, 32, 32]   (Stride 32)
        Returns:
            list of Tensors: [P2_out, P3_out, P4_out, P5_out]
        """
        p2, p3, p4, p5 = features

        # 1. Project channels into target dimensions
        p2_proj = self.proj_p2(p2)
        p3_proj = self.proj_p3(p3)
        p4_proj = self.proj_p4(p4)
        p5_proj = self.proj_p5(p5)

        # --- 2. Top-Down Adaptive Feature Fusion ---
        # P5 -> P4
        w_td4 = F.softmax(self.w_td_p4, dim=0)
        up_p5 = F.interpolate(self.up_p5_to_p4(p5_proj), size=p4_proj.shape[2:], mode='nearest')
        td_p4 = self.smooth_td_p4(w_td4[0] * p4_proj + w_td4[1] * up_p5)

        # P4 -> P3
        w_td3 = F.softmax(self.w_td_p3, dim=0)
        up_p4 = F.interpolate(self.up_p4_to_p3(td_p4), size=p3_proj.shape[2:], mode='nearest')
        td_p3 = self.smooth_td_p3(w_td3[0] * p3_proj + w_td3[1] * up_p4)

        # P3 -> P2
        w_td2 = F.softmax(self.w_td_p2, dim=0)
        up_p3 = F.interpolate(self.up_p3_to_p2(td_p3), size=p2_proj.shape[2:], mode='nearest')
        td_p2 = self.smooth_td_p2(w_td2[0] * p2_proj + w_td2[1] * up_p3)

        # --- 3. Bottom-Up Adaptive Feature Fusion ---
        # P2_out
        p2_out = td_p2 # [B, C2_out, 256, 256] (Stride 4)

        # P2 -> P3
        w_bu3 = F.softmax(self.w_bu_p3, dim=0)
        down_p2 = self.down_p2_to_p3(p2_out)
        p3_out = self.smooth_bu_p3(w_bu3[0] * td_p3 + w_bu3[1] * down_p2) # [B, C3_out, 128, 128] (Stride 8)

        # P3 -> P4
        w_bu4 = F.softmax(self.w_bu_p4, dim=0)
        down_p3 = self.down_p3_to_p4(p3_out)
        p4_out = self.smooth_bu_p4(w_bu4[0] * td_p4 + w_bu4[1] * down_p3) # [B, C4_out, 64, 64] (Stride 16)

        # P4 -> P5
        w_bu5 = F.softmax(self.w_bu_p5, dim=0)
        down_p4 = self.down_p4_to_p5(p4_out)
        p5_out = self.smooth_bu_p5(w_bu5[0] * p5_proj + w_bu5[1] * down_p4) # [B, C5_out, 32, 32] (Stride 32)

        return [p2_out, p3_out, p4_out, p5_out]


if __name__ == '__main__':
    # Simple Unit Test
    print("Testing AHSFPN...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p2 = torch.randn(2, 128, 256, 256, device=device)
    p3 = torch.randn(2, 256, 128, 128, device=device)
    p4 = torch.randn(2, 512, 64, 64, device=device)
    p5 = torch.randn(2, 1024, 32, 32, device=device)

    fpn = AHSFPN(in_channels=[128, 256, 512, 1024], out_channels=[128, 256, 512, 1024]).to(device)
    outs = fpn([p2, p3, p4, p5])

    print("Output shapes:")
    for i, out in enumerate(outs):
        print(f"P{i+2}_out: {out.shape}")
        assert out.shape == [p2, p3, p4, p5][i].shape, f"Shape mismatch at P{i+2}"

    sum(o.sum() for o in outs).backward()
    print("AHSFPN Unit Test PASSED!")
