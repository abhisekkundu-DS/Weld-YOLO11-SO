"""
End-to-End Training Pipeline Test for Weld-YOLO11-SO
Creates a temporary synthetic dataset and verifies model.train() execution.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pathlib
import sys
import shutil
import cv2
import numpy as np
import torch

# Add project root directory to python path
project_root = pathlib.Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.modules import register_custom_modules
register_custom_modules()

from ultralytics import YOLO


def create_dummy_dataset(dataset_dir):
    """Creates a minimal dummy dataset (YAML + images + labels) for training verification."""
    dataset_dir = pathlib.Path(dataset_dir)
    images_dir = dataset_dir / "images" / "train"
    labels_dir = dataset_dir / "labels" / "train"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Create 4 dummy 1024x1024 synthetic images with random bboxes
    for i in range(4):
        img_path = images_dir / f"sample_{i}.jpg"
        label_path = labels_dir / f"sample_{i}.txt"

        # Create dummy image
        img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        # Write dummy YOLO label: <class> <x_center> <y_center> <width> <height>
        # Class 0: Bad Weld, 1: Good Weld, 2: Defect
        cls_id = i % 3
        label_content = f"{cls_id} 0.5 0.5 0.2 0.2\n"
        label_path.write_text(label_content)

    # Create dataset yaml file
    data_yaml = dataset_dir / "data.yaml"
    data_content = f"""path: {dataset_dir.absolute()}
train: images/train
val: images/train

nc: 3
names:
  0: Bad Weld
  1: Good Weld
  2: Defect
"""
    data_yaml.write_text(data_content)
    return data_yaml


def test_training_pipeline():
    print("=" * 60)
    print("TESTING WELD-YOLO11-SO END-TO-END TRAINING PIPELINE")
    print("=" * 60)

    dummy_dir = project_root / "tests" / "dummy_dataset"
    try:
        # 1. Create Synthetic Test Dataset
        print("\n1. Generating synthetic test dataset...")
        data_yaml = create_dummy_dataset(dummy_dir)
        print(f"   Synthetic dataset YAML created at: {data_yaml}")

        # 2. Instantiate Weld-YOLO11-SO Model
        yaml_path = project_root / "models" / "weld_yolo11.yaml"
        print(f"\n2. Loading model from {yaml_path}...")
        model = YOLO(str(yaml_path))

        # 3. Execute 1 Epoch of Training
        print("\n3. Launching 1-epoch training run (batch=2, imgsz=1024)...")
        results = model.train(
            data=str(data_yaml),
            imgsz=1024,
            epochs=1,
            batch=2,
            workers=0,
            device='cpu', # Test on CPU for compatibility
            project=str(project_root / "tests" / "runs"),
            name="test_weld_yolo11",
            exist_ok=True,
            verbose=False
        )

        print("\n" + "=" * 60)
        print("SUCCESS: WELD-YOLO11-SO TRAINING PIPELINE PASSED 100% PERFECTLY!")
        print("=" * 60)

    finally:
        # Clean up temporary synthetic dataset and test runs
        if dummy_dir.exists():
            shutil.rmtree(dummy_dir, ignore_errors=True)
        runs_dir = project_root / "tests" / "runs"
        if runs_dir.exists():
            shutil.rmtree(runs_dir, ignore_errors=True)


if __name__ == '__main__':
    test_training_pipeline()
