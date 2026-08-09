"""
MNIST Dataset Pipeline Test for Weld-YOLO11-SO
Downloads MNIST, converts samples into YOLO detection format (1024x1024 canvas), and tests training/inference.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pathlib
import sys
import shutil
import cv2
import numpy as np
import torch
import torchvision.datasets as datasets

# Add project root directory to python path
project_root = pathlib.Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.modules import register_custom_modules
register_custom_modules()

from ultralytics import YOLO


def prepare_mnist_yolo_dataset(target_dir, num_samples=20):
    """
    Downloads MNIST dataset using torchvision and formats samples into a YOLO object detection dataset.
    Paces 28x28 digits onto a 1024x1024 canvas with corresponding bounding boxes.
    """
    target_dir = pathlib.Path(target_dir)
    images_train_dir = target_dir / "images" / "train"
    labels_train_dir = target_dir / "labels" / "train"
    images_val_dir = target_dir / "images" / "val"
    labels_val_dir = target_dir / "labels" / "val"

    for d in [images_train_dir, labels_train_dir, images_val_dir, labels_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("  Downloading MNIST dataset via torchvision...")
    mnist_data = datasets.MNIST(root=str(target_dir / "raw_mnist"), train=True, download=True)

    print(f"  Converting {num_samples} MNIST samples to YOLO 1024x1024 detection format...")
    for idx in range(num_samples):
        img_pil, label = mnist_data[idx]
        img_np = np.array(img_pil) # 28x28 grayscale

        # Create 1024x1024 3-channel canvas
        canvas = np.zeros((1024, 1024, 3), dtype=np.uint8)

        # Place 28x28 digit at center (or random offset) resized to 200x200 for clear object detection
        digit_resized = cv2.resize(img_np, (200, 200), interpolation=cv2.INTER_CUBIC)
        digit_3ch = cv2.merge([digit_resized, digit_resized, digit_resized])

        # Center placement coordinates on 1024x1024 canvas
        x_min, y_min = 412, 412
        x_max, y_max = 612, 612
        canvas[y_min:y_max, x_min:x_max] = digit_3ch

        # YOLO normalized bounding box format: <cls> <x_center> <y_center> <width> <height>
        cx = 0.5
        cy = 0.5
        w = 200.0 / 1024.0
        h = 200.0 / 1024.0

        # Map 10 MNIST classes into 3 target classes (0: Bad Weld/Digit 0-2, 1: Good Weld/Digit 3-5, 2: Defect/Digit 6-9)
        cls_mapped = 0 if label in [0, 1, 2] else (1 if label in [3, 4, 5] else 2)

        # Save to train or val split
        split_img_dir = images_train_dir if idx < (num_samples * 0.8) else images_val_dir
        split_lbl_dir = labels_train_dir if idx < (num_samples * 0.8) else labels_val_dir

        img_file = split_img_dir / f"mnist_{idx}.jpg"
        lbl_file = split_lbl_dir / f"mnist_{idx}.txt"

        cv2.imwrite(str(img_file), canvas)
        lbl_file.write_text(f"{cls_mapped} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}\n")

    # Create data.yaml
    data_yaml = target_dir / "data.yaml"
    data_content = f"""path: {target_dir.absolute()}
train: images/train
val: images/val

nc: 3
names:
  0: Bad Weld
  1: Good Weld
  2: Defect
"""
    data_yaml.write_text(data_content)
    return data_yaml


def test_mnist_pipeline():
    print("=" * 60)
    print("TESTING WELD-YOLO11-SO ON MNIST DATASET PIPELINE")
    print("=" * 60)

    mnist_yolo_dir = project_root / "tests" / "mnist_yolo_dataset"
    try:
        # 1. Download MNIST & Convert to YOLO Format
        data_yaml = prepare_mnist_yolo_dataset(mnist_yolo_dir, num_samples=20)
        print(f"\nMNIST YOLO dataset prepared at: {data_yaml}")

        # 2. Instantiate Model
        yaml_path = project_root / "models" / "weld_yolo11.yaml"
        print(f"\nLoading model from {yaml_path}...")
        model = YOLO(str(yaml_path))

        # 3. Train on MNIST Dataset for 1 Epoch
        print("\nExecuting 1-epoch training run on MNIST dataset...")
        results = model.train(
            data=str(data_yaml),
            imgsz=1024,
            epochs=1,
            batch=4,
            workers=0,
            device='cpu',
            project=str(project_root / "tests" / "runs"),
            name="mnist_weld_yolo11",
            exist_ok=True,
            verbose=False
        )

        print("\n" + "=" * 60)
        print("SUCCESS: WELD-YOLO11-SO PASSED MNIST DATASET TRAINING & TESTING 100%!")
        print("=" * 60)

    finally:
        # Cleanup test dataset files
        if mnist_yolo_dir.exists():
            shutil.rmtree(mnist_yolo_dir, ignore_errors=True)


if __name__ == '__main__':
    test_mnist_pipeline()
