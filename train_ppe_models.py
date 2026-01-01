#!/usr/bin/env python3
"""
Unified PPE Model Training Script for Research

Train multiple YOLO and RT-DETR models on Construction-PPE dataset for comparison.
Supports: YOLOv11 (n/s/m/l/x), YOLOv10 (m/l), RT-DETR-l

Usage:
    python train_ppe_models.py                    # Train all models
    python train_ppe_models.py --model yolo11l    # Train specific model
    python train_ppe_models.py --list             # List available models

Classes (10):
- Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest
- Person, Safety Cone, Safety Vest, machinery, vehicle
"""

import os
import argparse
import shutil
import torch
from pathlib import Path
from datetime import datetime

# Available models for PPE training
MODELS = {
    # YOLOv11 variants
    'yolo11n': {'base': 'yolo11n.pt', 'output': 'ppe_yolo11n_best.pt', 'type': 'yolo', 'batch': 16, 'params': '2.6M'},
    'yolo11s': {'base': 'yolo11s.pt', 'output': 'ppe_yolo11s_best.pt', 'type': 'yolo', 'batch': 12, 'params': '9.4M'},
    'yolo11m': {'base': 'yolo11m.pt', 'output': 'ppe_yolo11m_best.pt', 'type': 'yolo', 'batch': 8, 'params': '20.1M'},
    'yolo11l': {'base': 'yolo11l.pt', 'output': 'ppe_yolo11l_best.pt', 'type': 'yolo', 'batch': 4, 'params': '25.3M'},
    'yolo11x': {'base': 'yolo11x.pt', 'output': 'ppe_yolo11x_best.pt', 'type': 'yolo', 'batch': 2, 'params': '56.9M'},

    # YOLOv10 variants
    'yolov10m': {'base': 'yolov10m.pt', 'output': 'ppe_yolov10m_best.pt', 'type': 'yolo', 'batch': 8, 'params': '15.4M'},
    'yolov10l': {'base': 'yolov10l.pt', 'output': 'ppe_yolov10l_best.pt', 'type': 'yolo', 'batch': 4, 'params': '24.4M'},

    # RT-DETR (Transformer-based)
    'rtdetr-l': {'base': 'rtdetr-l.pt', 'output': 'ppe_rtdetr-l_best.pt', 'type': 'rtdetr', 'batch': 4, 'params': '32.0M'},
}

# Training configuration
CONFIG = {
    'dataset': 'construction-ppe.yaml',
    'epochs': 100,
    'imgsz': 640,
    'patience': 15,
    'workers': 8,
    'project': 'ppe_training',
}


def get_device():
    """Detect best available device."""
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        return device, gpu_memory
    elif torch.backends.mps.is_available():
        return 'mps', 8.0
    return 'cpu', 0


def train_model(model_name: str, force: bool = False):
    """Train a single model on PPE dataset."""
    if model_name not in MODELS:
        print(f"  ERROR: Unknown model '{model_name}'")
        return None

    config = MODELS[model_name]
    base_model = config['base']
    output_model = config['output']
    model_type = config['type']
    batch_size = config['batch']

    print("\n" + "=" * 70)
    print(f"  TRAINING: {model_name.upper()}")
    print("=" * 70)
    print(f"  Base:      {base_model} ({config['params']} params)")
    print(f"  Output:    {output_model}")
    print(f"  Type:      {model_type.upper()}")
    print(f"  Batch:     {batch_size}")
    print("=" * 70)

    output_path = Path(output_model)

    # Check if already trained
    if output_path.exists() and not force:
        print(f"\n  Already trained: {output_model}")
        print("  Use --force to retrain")
        return output_path

    # Check base model exists
    if not Path(base_model).exists():
        print(f"\n  ERROR: Base model not found: {base_model}")
        print("  Run download script first.")
        return None

    # Get device
    device, gpu_mem = get_device()

    # Adjust batch size for GPU memory
    if gpu_mem < 6:
        batch_size = max(1, batch_size // 2)
        print(f"  Reduced batch size to {batch_size} for low VRAM")

    # Load model
    print(f"\n  Loading {base_model}...")
    if model_type == 'rtdetr':
        from ultralytics import RTDETR
        model = RTDETR(base_model)
    else:
        from ultralytics import YOLO
        model = YOLO(base_model)

    # Train
    print("\n  Starting training...")
    run_name = f"{model_name}_ppe_{datetime.now().strftime('%Y%m%d')}"

    results = model.train(
        data=CONFIG['dataset'],
        epochs=CONFIG['epochs'],
        imgsz=CONFIG['imgsz'],
        batch=batch_size,
        patience=CONFIG['patience'],
        device=device,
        workers=CONFIG['workers'],
        project=CONFIG['project'],
        name=run_name,
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True,
        amp=True,
    )

    # Copy best weights
    best_weights = Path(f"{CONFIG['project']}/{run_name}/weights/best.pt")
    if best_weights.exists():
        shutil.copy(best_weights, output_path)
        print(f"\n  Saved: {output_model}")
        print(f"  Size:  {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        print("\n  WARNING: best.pt not found!")
        return None

    # Validate
    print("\n  Validating...")
    if model_type == 'rtdetr':
        from ultralytics import RTDETR
        model = RTDETR(str(output_path))
    else:
        from ultralytics import YOLO
        model = YOLO(str(output_path))

    metrics = model.val()

    print(f"\n  Results for {model_name}:")
    print(f"    mAP50:     {metrics.box.map50:.4f}")
    print(f"    mAP50-95:  {metrics.box.map:.4f}")

    return output_path


def list_models():
    """Display available models and their status."""
    print("\n" + "=" * 80)
    print("  AVAILABLE MODELS FOR PPE TRAINING")
    print("=" * 80)
    print(f"  {'Model':<12} {'Base File':<18} {'Params':<10} {'Batch':<6} {'Status'}")
    print("-" * 80)

    for name, config in MODELS.items():
        base_exists = Path(config['base']).exists()
        trained_exists = Path(config['output']).exists()

        if trained_exists:
            status = "Trained"
        elif base_exists:
            status = "Ready"
        else:
            status = "Need download"

        print(f"  {name:<12} {config['base']:<18} {config['params']:<10} {config['batch']:<6} {status}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Train PPE detection models')
    parser.add_argument('--model', '-m', type=str, help='Specific model to train')
    parser.add_argument('--list', '-l', action='store_true', help='List available models')
    parser.add_argument('--force', '-f', action='store_true', help='Force retrain even if exists')
    parser.add_argument('--all', '-a', action='store_true', help='Train all models')
    args = parser.parse_args()

    # Set working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    if args.list:
        list_models()
        return

    if args.model:
        train_model(args.model, force=args.force)
    elif args.all:
        print("\n  Training ALL models (this will take a while)...\n")
        for model_name in MODELS:
            try:
                train_model(model_name, force=args.force)
            except Exception as e:
                print(f"\n  ERROR training {model_name}: {e}")
                continue
    else:
        # Default: show help
        parser.print_help()
        print("\n  Examples:")
        print("    python train_ppe_models.py --list")
        print("    python train_ppe_models.py --model yolo11l")
        print("    python train_ppe_models.py --all")


if __name__ == '__main__':
    main()
