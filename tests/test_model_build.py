"""
Model Build & Forward Pass Test for Weld-YOLO11-SO
Tests loading models/weld_yolo11.yaml, module parsing, and dummy inference.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import pathlib
import torch

# Add project root directory to python path
project_root = pathlib.Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Register custom modules with Ultralytics framework
from models.modules import register_custom_modules
register_custom_modules()

from ultralytics import YOLO


def test_model_build():
    print("=" * 60)
    print("TESTING WELD-YOLO11-SO MODEL BUILD AND INFERENCE")
    print("=" * 60)

    yaml_path = project_root / "models" / "weld_yolo11.yaml"
    print(f"\n1. Loading model configuration from: {yaml_path}")
    assert yaml_path.exists(), f"Model YAML file not found at {yaml_path}"

    # Build YOLO model from YAML
    model = YOLO(str(yaml_path))
    print("\nSUCCESS: Model instantiated successfully without errors!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model moved to device: {device}")

    # 2. Test Training Mode Output (4 Multi-Scale Feature Outputs: P2, P3, P4, P5)
    model.model.train()
    x = torch.randn(1, 3, 1024, 1024, device=device)
    print(f"\n2. Testing Training Mode Forward Pass with input: {x.shape}...")
    
    train_preds = model.model(x)
    print(f"  Training output type: {type(train_preds)}")
    if isinstance(train_preds, list):
        print(f"  Number of detection scale outputs: {len(train_preds)}")
        for i, out in enumerate(train_preds):
            print(f"    Scale P{i+2} output shape: {out.shape}")
        assert len(train_preds) == 4, f"Expected 4 detection scales (P2, P3, P4, P5), got {len(train_preds)}"

    # 3. Test Evaluation Mode Output
    model.model.eval()
    print(f"\n3. Testing Evaluation Mode Forward Pass with input: {x.shape}...")
    with torch.no_grad():
        eval_preds = model.model(x)

    cat_preds = eval_preds[0] if isinstance(eval_preds, tuple) else eval_preds
    print(f"  Combined Inference Predictions shape: {cat_preds.shape}")
    
    # Verify shape: [B, 4 + nc, Total_Anchors] = [1, 7, 87040] for 1024x1024 input
    # Total anchors = 256^2 + 128^2 + 64^2 + 32^2 = 65536 + 16384 + 4096 + 1024 = 87040 anchors!
    expected_channels = 4 + 3 # 4 bbox coordinates + 3 classes (Bad Weld, Good Weld, Defect)
    expected_anchors = (256 * 256) + (128 * 128) + (64 * 64) + (32 * 32) # 87040

    assert cat_preds.shape[1] == expected_channels, f"Expected {expected_channels} channels (4 box + 3 classes), got {cat_preds.shape[1]}"
    assert cat_preds.shape[2] == expected_anchors, f"Expected {expected_anchors} anchors across P2-P5, got {cat_preds.shape[2]}"

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print(f"  - 4 Detection Scales: P2 (stride 4), P3 (stride 8), P4 (stride 16), P5 (stride 32)")
    print(f"  - Total Anchors at 1024x1024 input: {cat_preds.shape[2]}")
    print(f"  - Channel Outputs: {cat_preds.shape[1]} (4 box coords + 3 classes)")
    print("=" * 60)


if __name__ == '__main__':
    test_model_build()
