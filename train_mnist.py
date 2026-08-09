"""
Weld-YOLO11-SO: Train 3 Epochs on MNIST Dataset
Command to run in terminal:
    python train_mnist.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pathlib
import sys
import cv2
import numpy as np
import torchvision.datasets as datasets

# Add project root directory to python path
project_root = pathlib.Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Register custom modules with Ultralytics parser
from models.modules import register_custom_modules
register_custom_modules()

from ultralytics import YOLO


# Override default torchvision MNIST mirrors to reliable AWS / Google storage mirrors
datasets.MNIST.mirrors = [
    'https://storage.googleapis.com/cvdf-datasets/mnist/',
    'https://ossci-datasets.s3.amazonaws.com/mnist/'
]


def create_synthetic_digit_image(digit):
    """Generates a clean synthetic digit image (28x28) as fallback."""
    img = np.zeros((28, 28), dtype=np.uint8)
    cv2.putText(img, str(digit), (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    return img


def prepare_mnist_yolo_dataset(target_dir, num_samples=100):
    """
    Downloads MNIST dataset via torchvision or generates synthetic digit samples,
    and formats them into 1024x1024 YOLO bounding box detection format.
    """
    target_dir = pathlib.Path(target_dir)
    images_train_dir = target_dir / "images" / "train"
    labels_train_dir = target_dir / "labels" / "train"
    images_val_dir = target_dir / "images" / "val"
    labels_val_dir = target_dir / "labels" / "val"

    for d in [images_train_dir, labels_train_dir, images_val_dir, labels_val_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"1. Preparing {num_samples} MNIST digit samples in YOLO format...")
    mnist_samples = []

    try:
        print("   Downloading MNIST dataset...")
        mnist_data = datasets.MNIST(root=str(target_dir / "raw_mnist"), train=True, download=True)
        for idx in range(min(num_samples, len(mnist_data))):
            img_pil, label = mnist_data[idx]
            mnist_samples.append((np.array(img_pil), label))
    except Exception as e:
        print(f"   Note: Torchvision download fallback triggered ({e}). Generating synthetic digit dataset...")
        for idx in range(num_samples):
            label = idx % 10
            img_np = create_synthetic_digit_image(label)
            mnist_samples.append((img_np, label))

    for idx, (img_np, label) in enumerate(mnist_samples):
        # Create 1024x1024 3-channel canvas
        canvas = np.zeros((1024, 1024, 3), dtype=np.uint8)

        # Resize 28x28 digit to 200x200 for small object detection training
        digit_resized = cv2.resize(img_np, (200, 200), interpolation=cv2.INTER_CUBIC)
        digit_3ch = cv2.merge([digit_resized, digit_resized, digit_resized])

        # Place digit in center (412, 412) to (612, 612)
        canvas[412:612, 412:612] = digit_3ch

        # YOLO normalized bounding box format: <class> <x_center> <y_center> <width> <height>
        cx, cy = 0.5, 0.5
        w, h = 200.0 / 1024.0, 200.0 / 1024.0

        # Map 10 MNIST classes to 3 target classes:
        # 0: Bad Weld (digits 0-2), 1: Good Weld (digits 3-5), 2: Defect (digits 6-9)
        cls_mapped = 0 if label in [0, 1, 2] else (1 if label in [3, 4, 5] else 2)

        is_train = idx < int(len(mnist_samples) * 0.8)
        split_img_dir = images_train_dir if is_train else images_val_dir
        split_lbl_dir = labels_train_dir if is_train else labels_val_dir

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


def run_3_epochs_mnist():
    print("=" * 70)
    print("WELD-YOLO11-SO: 3-EPOCH TRAINING ON MNIST DATASET")
    print("=" * 70)

    mnist_dir = project_root / "mnist_dataset"
    data_yaml = prepare_mnist_yolo_dataset(mnist_dir, num_samples=100)
    print(f"\nDataset Ready at: {data_yaml}")

    # Load Weld-YOLO11-SO Model Architecture
    yaml_path = project_root / "models" / "weld_yolo11.yaml"
    print(f"\n2. Loading Weld-YOLO11-SO model from: {yaml_path}")
    model = YOLO(str(yaml_path))

    # Run 3 Epochs of Training
    print("\n3. Starting 3-Epoch Training Run (imgsz=1024, batch=4)...")
    results = model.train(
        data=str(data_yaml),
        imgsz=1024,
        epochs=3,
        batch=4,
        workers=0,
        project="runs",
        name="mnist_weld_yolo11_3epochs",
        exist_ok=True
    )

    print("\n" + "=" * 70)
    print("3-EPOCH TRAINING COMPLETED SUCCESSFULLY!")
    print("Model weights saved to: runs/mnist_weld_yolo11_3epochs/weights/best.pt")
    print("=" * 70)


if __name__ == '__main__':
    run_3_epochs_mnist()
