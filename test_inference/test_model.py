"""
Weld-YOLO11-SO: Standalone Test Inference Script
Loads trained weights (best_70_epochs.pt), runs detection on input_images/,
and saves annotated prediction images to output_images/.
"""

import os
import sys
import cv2
import pathlib
from pathlib import Path

# 1. Prevent OpenMP library collision on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 2. Add repository root directory to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent if (current_dir.parent / "models" / "weld_yolo11.yaml").exists() else current_dir

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 3. Register custom Weld-YOLO11-SO modules with Ultralytics
from models.modules import register_custom_modules
register_custom_modules()

from ultralytics import YOLO


def run_test_inference():
    print("=" * 70)
    print("WELD-YOLO11-SO: STANDALONE MODEL INFERENCE TEST")
    print("=" * 70)

    # 4. Locate model weights
    weights_path = project_root / "trained_models" / "best_70_epochs.pt"
    if not weights_path.exists():
        weights_path = project_root / "runs" / "weld_training" / "weld_yolo11_so" / "weights" / "best.pt"

    print(f"\n1. Loading Model Weights from: {weights_path}")
    if not weights_path.exists():
        print(f"❌ Error: Weights file not found at {weights_path}")
        return

    model = YOLO(str(weights_path))
    print("   Model loaded successfully!")

    # 5. Define input and output directories
    input_dir = current_dir / "input_images"
    output_dir = current_dir / "output_images"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    input_images = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in img_exts])

    if not input_images:
        print(f"\n❌ No input images found in: {input_dir}")
        print("   Please place test images (.jpg/.png) inside input_images/ folder.")
        return

    print(f"\n2. Found {len(input_images)} test images in input_images/ directory:")
    for img in input_images:
        print(f"   - {img.name}")

    print("\n3. Running Inference and Generating Output Images...")
    print("-" * 70)

    total_detections = 0
    for idx, img_path in enumerate(input_images, 1):
        # Run YOLO prediction
        results = model.predict(source=str(img_path), conf=0.25, imgsz=1024, verbose=False)

        for res in results:
            boxes = res.boxes
            num_dets = len(boxes)
            total_detections += num_dets

            print(f"[{idx}/{len(input_images)}] Image: {img_path.name:<25} | Detections: {num_dets}")

            for b in boxes:
                cls_id = int(b.cls[0])
                cls_name = model.names.get(cls_id, str(cls_id))
                conf = float(b.conf[0])
                bbox = [round(x) for x in b.xyxy[0].tolist()]
                print(f"       -> Class: {cls_name:<10} | Confidence: {conf:.2%} | Bounding Box: {bbox}")

            # Save annotated image
            annotated_frame = res.plot()
            out_image_path = output_dir / f"result_{img_path.stem}.jpg"
            cv2.imwrite(str(out_image_path), annotated_frame)
            print(f"       Annotated Output Saved to: output_images/{out_image_path.name}\n")

    print("=" * 70)
    print(f"SUCCESS: Inference Completed!")
    print(f"Processed Images  : {len(input_images)}")
    print(f"Total Detections  : {total_detections}")
    print(f"Output Directory  : {output_dir.resolve()}")
    print("=" * 70)


if __name__ == '__main__':
    run_test_inference()
