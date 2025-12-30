# PPE Detection System

Enhanced PPE (Personal Protective Equipment) detection system using YOLOv11 with fish-eye lens correction and dual reporting.

## Features

- **YOLOv11 Large Model**: Trained on Construction-PPE dataset for maximum accuracy
- **Fish-Eye Lens Correction**: Corrects distortion from 360° and ceiling cameras
- **Class-Specific Confidence Thresholds**: Optimized thresholds per detection class
- **Adaptive PPE Association**: Smart margin calculation based on person size
- **Dual Reporting**: Developer (technical) and Client (business) reports
- **CUDA Optimized**: Configured for NVIDIA RTX 4060 (8GB VRAM)

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OR .venv\Scripts\activate  # Windows

# Install requirements
pip install ultralytics opencv-python pandas numpy reportlab tqdm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
```

### 2. Train YOLOv11 Model (First Time Only)

```bash
python train_yolo11l.py
```

This will:
- Download the Construction-PPE dataset (~178MB)
- Train YOLOv11l for 100 epochs
- Save the trained model as `ppe_yolo11l_best.pt`

**Expected time**: ~1-2 hours on RTX 4060

### 3. Download Images from AWS S3

Place your images in:
```
s3_images/
├── left/     # Left camera images
├── right/    # Right camera images
└── down/     # Down camera images
```

### 4. Run Detection

```bash
python ppe_report.py
```

This generates:
- Annotated images in `report_X/`
- Detection CSV: `all_detections.csv`
- HTML report: `report.html`
- JSON summary: `summary.json`

### 5. Generate Reports

```bash
# Both developer and client reports
python generate_reports.py --type both

# Developer report only (technical details)
python generate_reports.py --type developer

# Client report only (business summary)
python generate_reports.py --type client
```

## File Structure

```
├── train_yolo11l.py          # Train YOLOv11 on PPE dataset
├── ppe_report.py             # Main detection pipeline
├── generate_reports.py       # Report generation entry point
├── lens_correction.py        # Fish-eye correction module
├── camera_calibration.json   # Camera calibration parameters
├── reporting/                # Report generation module
│   ├── __init__.py
│   ├── report_config.py
│   ├── data_processor.py
│   ├── metrics_calculator.py
│   ├── developer_report.py
│   └── client_report.py
└── s3_images/                # Input images (from AWS)
    ├── left/
    ├── right/
    └── down/
```

## Configuration

### Detection Settings (ppe_report.py)

```python
CONFIG = {
    'model_path': 'ppe_yolo11l_best.pt',
    'base_confidence': 0.15,
    'lens_correction_enabled': True,
    'apply_clahe': True,           # Contrast enhancement
    'normalize_brightness': True,  # Brightness normalization
}

# Class-specific confidence thresholds
CLASS_CONFIDENCE_THRESHOLDS = {
    'person': 0.20,        # Lower for better recall
    'helmet': 0.35,
    'safety-vest': 0.30,
    'NO-Hardhat': 0.50,    # Higher for violations
}
```

### Camera Calibration (camera_calibration.json)

Adjust distortion coefficients if needed:
- **k1**: Negative for barrel distortion (fish-eye)
- **k2**: Fine-tuning coefficient
- **balance**: 0.0 = crop black borders, 1.0 = keep full image

## GPU Optimization

### For NVIDIA RTX 4060 (8GB):
- Batch size: 16
- Workers: 8
- Mixed precision (AMP): Enabled

### For other GPUs:
Edit `train_yolo11l.py`:
```python
CONFIG = {
    'batch': 8,   # Reduce for less VRAM
    'workers': 4, # Reduce for less CPU
}
```

## Detected Classes

| Class | Description |
|-------|-------------|
| Person | Workers in the scene |
| Hardhat | Safety helmet detected |
| Safety Vest | High-visibility vest |
| NO-Hardhat | Missing helmet violation |
| NO-Safety Vest | Missing vest violation |
| Mask | Face mask |
| Safety Cone | Traffic cone |
| machinery | Construction equipment |
| vehicle | Vehicles |

## Output Reports

### Developer Report (PDF)
- Model performance metrics
- Confidence score analysis
- Per-class breakdown
- Low-confidence detections
- Full detection CSV

### Client Report (PDF)
- Compliance status (Green/Yellow/Red)
- Overall compliance rate
- Key statistics
- Recommendations

## Troubleshooting

### CUDA Not Detected
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory
Reduce batch size in `train_yolo11l.py` or `ppe_report.py`

### Model Not Found
Run `python train_yolo11l.py` first, or use fallback model:
```python
CONFIG['model_path'] = 'ppe_yolov8_best.pt'  # Fallback
```

## License

MIT License
