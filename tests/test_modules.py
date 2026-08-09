"""
Unit Tests for Weld-YOLO11-SO Custom Modules
Tests forward pass, shape preservation, backward pass, CUDA/CPU compatibility.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import pathlib
import torch

# Add project root directory to python import path
project_root = pathlib.Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.modules.wavelet import WaveletBlock
from models.modules.weldsimam import WeldSimAM
from models.modules.dysample import DySample
from models.modules.ahsfpn import AHSFPN


def test_wavelet_block(device):
    print(f"\n[1/4] Testing WaveletBlock on {device}...")
    x = torch.randn(2, 128, 80, 80, device=device, requires_grad=True)
    block = WaveletBlock(c1=128, c2=128).to(device)
    out = block(x)

    print(f"  Input: {x.shape} -> Output: {out.shape}")
    assert out.shape == (2, 128, 80, 80), f"WaveletBlock output shape mismatch! Got {out.shape}"
    
    # Backward pass & gradient check
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0, "WaveletBlock gradient check failed!"
    print("  WaveletBlock PASSED!")


def test_weld_simam(device):
    print(f"\n[2/4] Testing WeldSimAM on {device}...")
    x = torch.randn(2, 128, 80, 80, device=device, requires_grad=True)
    simam = WeldSimAM(c1=128, c2=128).to(device)
    out = simam(x)

    print(f"  Input: {x.shape} -> Output: {out.shape}")
    assert out.shape == (2, 128, 80, 80), f"WeldSimAM output shape mismatch! Got {out.shape}"

    # Backward pass & gradient check
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0, "WeldSimAM gradient check failed!"
    print("  WeldSimAM PASSED!")


def test_dysample(device):
    print(f"\n[3/4] Testing DySample on {device}...")
    x = torch.randn(2, 128, 40, 40, device=device, requires_grad=True)
    dysampler = DySample(c1=128, c2=128, scale=2).to(device)
    out = dysampler(x)

    print(f"  Input: {x.shape} -> Output: {out.shape}")
    assert out.shape == (2, 128, 80, 80), f"DySample output shape mismatch! Got {out.shape}"

    # Backward pass & gradient check
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0, "DySample gradient check failed!"
    print("  DySample PASSED!")


def test_ahsfpn(device):
    print(f"\n[4/4] Testing AHSFPN on {device}...")
    p2 = torch.randn(2, 128, 256, 256, device=device, requires_grad=True)
    p3 = torch.randn(2, 256, 128, 128, device=device, requires_grad=True)
    p4 = torch.randn(2, 512, 64, 64, device=device, requires_grad=True)
    p5 = torch.randn(2, 1024, 32, 32, device=device, requires_grad=True)

    fpn = AHSFPN(in_channels=[128, 256, 512, 1024], out_channels=[128, 256, 512, 1024]).to(device)
    outs = fpn([p2, p3, p4, p5])

    expected_shapes = [(2, 128, 256, 256), (2, 256, 128, 128), (2, 512, 64, 64), (2, 1024, 32, 32)]
    for i, out in enumerate(outs):
        print(f"  P{i+2}_out shape: {out.shape}")
        assert out.shape == expected_shapes[i], f"AHSFPN output shape mismatch at P{i+2}!"

    # Backward pass & gradient check
    loss = sum(o.sum() for o in outs)
    loss.backward()
    assert p2.grad is not None and p2.grad.abs().sum() > 0, "AHSFPN gradient check failed!"
    print("  AHSFPN PASSED!")


def main():
    print("=" * 60)
    print("RUNNING WELD-YOLO11-SO CUSTOM MODULES UNIT TESTS")
    print("=" * 60)

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for dev in devices:
        device = torch.device(dev)
        print(f"\n---> Testing on Device: {device}")
        test_wavelet_block(device)
        test_weld_simam(device)
        test_dysample(device)
        test_ahsfpn(device)

    print("\n" + "=" * 60)
    print("ALL UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == '__main__':
    main()
