#!/usr/bin/env python3
"""
Train YOLOv11l (Large) on Construction-PPE Dataset

This script trains the largest YOLOv11 model on the PPE dataset for maximum accuracy.
The construction-ppe dataset auto-downloads from Ultralytics (~178MB).

Classes (10):
- Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest
- Person, Safety Cone, Safety Vest, machinery, vehicle

Output: ppe_yolo11l_best.pt
"""

import os
import shutil
import torch
from pathlib import Path
from ultralytics import YOLO

# Configuration
CONFIG = {
    'base_model': 'yolo11l.pt',           # Large model (51MB, 25.3M params)
    'trained_model': 'ppe_yolo11l_best.pt',
    'dataset': 'construction-ppe.yaml',   # Auto-downloads from Ultralytics

    # Training parameters - OPTIMIZED FOR NVIDIA RTX 4060 (8GB VRAM)
    'epochs': 100,          # More epochs for better convergence
    'imgsz': 640,           # Image size
    'batch': 16,            # Batch size optimized for 4060 8GB VRAM
    'patience': 15,         # Early stopping patience
    'workers': 8,           # Data loader workers

    # Project settings
    'project': 'ppe_training',
    'name': 'yolo11l_ppe',
}

def main():
    # Set working directory (update this for your machine)
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Detect device - PREFER CUDA for NVIDIA GPUs
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU Detected: {gpu_name} ({gpu_memory:.1f}GB)")
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print("=" * 70)
    print("  YOLO11L PPE TRAINING")
    print("=" * 70)
    print(f"\n  Base Model:     {CONFIG['base_model']} (25.3M params)")
    print(f"  Output:         {CONFIG['trained_model']}")
    print(f"  Dataset:        {CONFIG['dataset']}")
    print(f"  Device:         {device}")
    print(f"\n  Training Config:")
    print(f"    Epochs:       {CONFIG['epochs']}")
    print(f"    Image Size:   {CONFIG['imgsz']}")
    print(f"    Batch Size:   {CONFIG['batch']}")
    print(f"    Patience:     {CONFIG['patience']}")
    print("\n" + "=" * 70)

    trained_path = Path(CONFIG['trained_model'])

    if trained_path.exists():
        print(f"\n  Trained model already exists: {CONFIG['trained_model']}")
        print("  Delete it to retrain, or use the existing model.")
        return

    # Load base model
    print("\n  Loading base model...")
    model = YOLO(CONFIG['base_model'])

    # Train
    print("\n  Starting training...")
    print("  (Dataset will auto-download on first run ~178MB)\n")

    results = model.train(
        data=CONFIG['dataset'],
        epochs=CONFIG['epochs'],
        imgsz=CONFIG['imgsz'],
        batch=CONFIG['batch'],
        patience=CONFIG['patience'],
        device=device,
        workers=CONFIG.get('workers', 8),
        project=CONFIG['project'],
        name=CONFIG['name'],
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True,
        amp=True,  # Mixed precision for faster training on CUDA
    )

    # Copy best weights to project root
    best_weights = Path(f"{CONFIG['project']}/{CONFIG['name']}/weights/best.pt")
    if best_weights.exists():
        shutil.copy(best_weights, trained_path)
        print(f"\n  Saved: {CONFIG['trained_model']}")
        print(f"  Size:  {trained_path.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print("\n  WARNING: best.pt not found in training output!")

    # Validate
    print("\n  Validating model...")
    model = YOLO(str(trained_path))
    metrics = model.val()

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n  Model saved: {CONFIG['trained_model']}")
    print(f"  mAP50:       {metrics.box.map50:.3f}")
    print(f"  mAP50-95:    {metrics.box.map:.3f}")
    print("\n  Classes:")
    for i, name in enumerate(model.names.values()):
        print(f"    {i}: {name}")


if __name__ == '__main__':
    main()
