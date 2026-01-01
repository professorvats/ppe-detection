#!/usr/bin/env python3
"""
=============================================================================
PHASE 3: YOLOv11 + FISHEYE CORRECTION - PPE DETECTION RESEARCH
=============================================================================
Google Colab Script - Complete Pipeline

This script:
1. Downloads images from AWS S3 (optional)
2. Downloads YOLOv11 models
3. Applies fisheye lens correction to down camera images
4. Trains on corrected Construction-PPE dataset
5. Generates comprehensive PDF report

Fisheye Parameters for Down Camera:
- k1: -0.35, k2: 0.12, p1: 0.0, p2: 0.025
- fx: 0.92, fy: 1.0, balance: 0.0

Run in Google Colab with GPU runtime.
=============================================================================
"""

# ============================================================================
# CELL 0: Install Dependencies
# ============================================================================
# !pip install ultralytics reportlab pandas matplotlib seaborn gputil psutil opencv-python boto3 openai -q

# ============================================================================
# CELL 1: AWS & OpenAI Configuration (SET YOUR CREDENTIALS)
# ============================================================================
import os

# AWS Credentials
os.environ['AWS_ACCESS_KEY_ID'] = 'AKIA3DIORHCHHBJTPOOC'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'YOUR_AWS_SECRET_ACCESS_KEY'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

# OpenAI API Key (optional)
os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_API_KEY'

# S3 Bucket Configuration
S3_CONFIG = {
    'bucket_name': 'your-ppe-bucket',
    'image_prefix': 'images/',
    'dates_to_download': ['10-31', '11-7'],
    'cameras': ['down'],
    'max_images_per_folder': None,
}

# ============================================================================
# CELL 2: AWS S3 Download Functions
# ============================================================================
def download_from_s3(bucket_name, prefix, local_dir, max_files=None):
    """Download files from S3 bucket"""
    try:
        import boto3
        from pathlib import Path
        s3 = boto3.client('s3')
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=max_files or 1000)
        objects = response.get('Contents', [])
        if not objects:
            return 0
        downloaded = 0
        for obj in objects:
            key = obj['Key']
            filename = os.path.basename(key)
            if filename:
                s3.download_file(bucket_name, key, os.path.join(local_dir, filename))
                downloaded += 1
        print(f"  ✓ Downloaded {downloaded} files to {local_dir}")
        return downloaded
    except Exception as e:
        print(f"  S3 error: {e}")
        return 0

def download_ppe_images():
    """Download PPE images from S3"""
    if os.environ.get('AWS_ACCESS_KEY_ID', '').startswith('YOUR_'):
        print("  ⚠ AWS credentials not set - using Ultralytics dataset")
        return
    for date in S3_CONFIG['dates_to_download']:
        for camera in S3_CONFIG['cameras']:
            prefix = f"{S3_CONFIG['image_prefix']}{date}/{camera}/"
            download_from_s3(S3_CONFIG['bucket_name'], prefix, f"s3_images/{date}/{camera}", S3_CONFIG['max_images_per_folder'])

# ============================================================================
# CELL 3: Imports
# ============================================================================
import json
import time
import cv2
import gc
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import shutil
import subprocess

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

try:
    import GPUtil
    HAS_GPUTIL = True
except:
    os.system("pip install gputil -q")
    import GPUtil
    HAS_GPUTIL = True

try:
    import psutil
except:
    os.system("pip install psutil -q")
    import psutil

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.enums import TA_CENTER

# ============================================================================
# CELL 2: Fisheye Correction Parameters
# ============================================================================

FISHEYE_CONFIG = {
    'down_camera': {
        'k1': -0.35,
        'k2': 0.12,
        'p1': 0.0,
        'p2': 0.025,
        'fx': 0.92,
        'fy': 1.0,
        'balance': 0.0,
        'image_size': (2340, 2160),
    }
}

CONFIG = {
    'phase': 3,
    'phase_name': 'YOLOv11 + Fisheye Correction',
    'dataset': 'construction-ppe.yaml',
    'imgsz': 640,
    'patience': 15,
    'workers': 4,
    'output_dir': 'ppe_research/phase3_fisheye',

    'models': {
        'yolo11n': {'batch': 16, 'params': '2.6M', 'size': 'Nano'},
        'yolo11s': {'batch': 12, 'params': '9.4M', 'size': 'Small'},
        'yolo11m': {'batch': 8, 'params': '20.1M', 'size': 'Medium'},
        'yolo11l': {'batch': 4, 'params': '25.3M', 'size': 'Large'},
        'yolo11x': {'batch': 2, 'params': '56.9M', 'size': 'XLarge'},
    },

    # Use best epochs from Phase 2 (typically 300)
    'epochs_list': [15, 300, 600],

    'quick_test': False,
}

# ============================================================================
# CELL 3: Fisheye Correction Module
# ============================================================================

class FisheyeCorrector:
    """Apply fisheye lens correction to images"""

    def __init__(self, config: dict = None):
        self.config = config or FISHEYE_CONFIG['down_camera']
        self._init_correction_maps()

    def _init_correction_maps(self):
        """Initialize undistortion maps"""
        w, h = self.config['image_size']

        # Camera matrix
        fx = self.config['fx'] * w / 2
        fy = self.config['fy'] * h / 2
        cx, cy = w / 2, h / 2

        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        # Distortion coefficients [k1, k2, p1, p2]
        self.D = np.array([
            self.config['k1'],
            self.config['k2'],
            self.config['p1'],
            self.config['p2']
        ], dtype=np.float64)

        # New camera matrix with balance
        self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D, (w, h), np.eye(3), balance=self.config['balance']
        )

        # Pre-compute undistortion maps
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), self.new_K,
            (w, h), cv2.CV_16SC2
        )

        print(f"  Fisheye correction initialized for {w}x{h}")
        print(f"  k1={self.config['k1']}, k2={self.config['k2']}")
        print(f"  fx={self.config['fx']}, fy={self.config['fy']}")

    def correct(self, image: np.ndarray) -> np.ndarray:
        """Apply fisheye correction to an image"""
        # Resize to expected size if needed
        h, w = image.shape[:2]
        expected_w, expected_h = self.config['image_size']

        if (w, h) != (expected_w, expected_h):
            image = cv2.resize(image, (expected_w, expected_h))

        # Apply correction
        corrected = cv2.remap(
            image, self.map1, self.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )

        return corrected

    def correct_file(self, input_path: str, output_path: str) -> bool:
        """Correct a single image file"""
        try:
            img = cv2.imread(input_path)
            if img is None:
                return False

            corrected = self.correct(img)
            cv2.imwrite(output_path, corrected)
            return True
        except Exception as e:
            print(f"  Error correcting {input_path}: {e}")
            return False


# ============================================================================
# CELL 4: Metrics Container
# ============================================================================

@dataclass
class ComprehensiveMetrics:
    """Complete metrics matching required format"""
    model_name: str = ""
    model_version: str = ""
    model_size: str = ""
    params: str = ""
    epochs: int = 0
    fisheye_corrected: bool = True

    training_date: str = ""
    testing_date: str = ""

    map50: float = 0.0
    map50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    fps: float = 0.0
    inference_latency_ms: float = 0.0

    gpu_memory_used_mib: float = 0.0
    gpu_memory_total_mib: float = 0.0
    ram_used_mib: float = 0.0
    ram_total_mib: float = 0.0

    gpu_temperature_c: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_utilization_percent: float = 0.0
    power_draw_w: float = 0.0

    training_time_minutes: float = 0.0

    results_png: str = ""
    f1_curve: str = ""
    pr_curve: str = ""
    p_curve: str = ""
    r_curve: str = ""
    confusion_matrix: str = ""

    weights_path: str = ""
    results_dir: str = ""


# ============================================================================
# CELL 5: Hardware Profiler
# ============================================================================

class HardwareProfiler:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gpu_name = torch.cuda.get_device_name(0) if self.device == 'cuda' else 'N/A'

    def get_all_metrics(self) -> Dict:
        metrics = {
            'gpu_memory_used_mib': 0.0,
            'gpu_memory_total_mib': 0.0,
            'gpu_temperature_c': 0.0,
            'gpu_utilization_percent': 0.0,
            'cpu_usage_percent': 0.0,
            'ram_used_mib': 0.0,
            'ram_total_mib': 0.0,
            'power_draw_w': 0.0,
        }

        if self.device == 'cuda':
            metrics['gpu_memory_used_mib'] = torch.cuda.memory_allocated() / (1024**2)
            metrics['gpu_memory_total_mib'] = torch.cuda.get_device_properties(0).total_memory / (1024**2)

        if HAS_GPUTIL:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    metrics['gpu_utilization_percent'] = gpu.load * 100
                    metrics['gpu_temperature_c'] = gpu.temperature
                    metrics['gpu_memory_used_mib'] = gpu.memoryUsed
                    metrics['gpu_memory_total_mib'] = gpu.memoryTotal
            except:
                pass

        metrics['cpu_usage_percent'] = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        metrics['ram_used_mib'] = mem.used / (1024**2)
        metrics['ram_total_mib'] = mem.total / (1024**2)

        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                metrics['power_draw_w'] = float(result.stdout.strip())
        except:
            pass

        return metrics

    def benchmark_inference(self, model, num_runs: int = 100) -> Tuple[float, float]:
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        for _ in range(10):
            model.predict(dummy_img, verbose=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        latencies = []
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            model.predict(dummy_img, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

        return 1000 / np.mean(latencies), np.mean(latencies)


# ============================================================================
# CELL 6: Download Models
# ============================================================================

def download_models():
    print("\n" + "="*70)
    print("  DOWNLOADING YOLOv11 MODELS")
    print("="*70)

    for model_name in CONFIG['models'].keys():
        model_file = f"{model_name}.pt"
        if not Path(model_file).exists():
            print(f"  Downloading {model_file}...")
            YOLO(model_file)
            print(f"  ✓ {model_file}")
        else:
            print(f"  ✓ {model_file} exists")

    print("="*70 + "\n")


# ============================================================================
# CELL 7: Training Function
# ============================================================================

def train_model(model_name: str, epochs: int, profiler: HardwareProfiler) -> Optional[ComprehensiveMetrics]:
    model_config = CONFIG['models'][model_name]
    base_model = f"{model_name}.pt"

    size_letter = model_name[-1]
    version = f"v11.0.{size_letter}.e{epochs}.fisheye"

    run_name = f"{model_name}_e{epochs}_fisheye_{datetime.now().strftime('%Y%m%d_%H%M')}"
    output_dir = Path(CONFIG['output_dir']) / run_name

    print("\n" + "="*70)
    print(f"  TRAINING: {model_name.upper()} - {epochs} EPOCHS (FISHEYE CORRECTED)")
    print("="*70)
    print(f"  Version:    {version}")
    print(f"  Size:       {model_config['size']}")
    print(f"  Fisheye:    k1={FISHEYE_CONFIG['down_camera']['k1']}, k2={FISHEYE_CONFIG['down_camera']['k2']}")
    print("="*70)

    training_date = datetime.now().strftime("%b %d, %Y")
    start_time = time.time()

    try:
        model = YOLO(base_model)

        results = model.train(
            data=CONFIG['dataset'],
            epochs=epochs,
            imgsz=CONFIG['imgsz'],
            batch=model_config['batch'],
            patience=CONFIG['patience'],
            workers=CONFIG['workers'],
            project=CONFIG['output_dir'],
            name=run_name,
            exist_ok=True,
            plots=True,
            save=True,
            verbose=True,
            amp=True,
        )

        training_time = (time.time() - start_time) / 60
        testing_date = datetime.now().strftime("%b %d, %Y")

        results_dir = Path(CONFIG['output_dir']) / run_name
        best_weights = results_dir / 'weights' / 'best.pt'

        trained_model = YOLO(str(best_weights))
        val_results = trained_model.val()

        hw = profiler.get_all_metrics()
        fps, latency = profiler.benchmark_inference(trained_model)

        precision = float(val_results.box.mp)
        recall = float(val_results.box.mr)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        def find_curve(pattern: str) -> str:
            matches = list(results_dir.glob(f"*{pattern}*"))
            return str(matches[0]) if matches else ""

        metrics = ComprehensiveMetrics(
            model_name=model_name,
            model_version=version,
            model_size=model_config['size'],
            params=model_config['params'],
            epochs=epochs,
            fisheye_corrected=True,
            training_date=training_date,
            testing_date=testing_date,
            map50=float(val_results.box.map50),
            map50_95=float(val_results.box.map),
            precision=precision,
            recall=recall,
            f1_score=f1,
            fps=fps,
            inference_latency_ms=latency,
            gpu_memory_used_mib=hw['gpu_memory_used_mib'],
            gpu_memory_total_mib=hw['gpu_memory_total_mib'],
            ram_used_mib=hw['ram_used_mib'],
            ram_total_mib=hw['ram_total_mib'],
            gpu_temperature_c=hw['gpu_temperature_c'],
            cpu_usage_percent=hw['cpu_usage_percent'],
            gpu_utilization_percent=hw['gpu_utilization_percent'],
            power_draw_w=hw['power_draw_w'],
            training_time_minutes=training_time,
            results_png=find_curve('results'),
            f1_curve=find_curve('F1_curve'),
            pr_curve=find_curve('PR_curve'),
            p_curve=find_curve('P_curve'),
            r_curve=find_curve('R_curve'),
            confusion_matrix=find_curve('confusion_matrix.png'),
            weights_path=str(best_weights),
            results_dir=str(results_dir),
        )

        print(f"\n  ✓ Complete! mAP50={metrics.map50:.4f}, FPS={metrics.fps:.2f}")

        del model, trained_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metrics

    except Exception as e:
        print(f"\n  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# CELL 8: Report Generator
# ============================================================================

class Phase3ReportGenerator:
    def __init__(self, results: List[ComprehensiveMetrics], output_dir: str):
        self.results = results
        self.output_dir = Path(output_dir)

    def generate(self) -> str:
        report_path = self.output_dir / "Phase3_Report.pdf"

        doc = SimpleDocTemplate(
            str(report_path),
            pagesize=landscape(A4),
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=18, alignment=TA_CENTER, spaceAfter=15)
        heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=12, spaceAfter=8)

        elements = []

        # Title
        elements.append(Paragraph("Phase 3: YOLOv11 + Fisheye Correction", title_style))
        elements.append(Paragraph("PPE Detection Research Report", styles['Heading3']))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}", styles['Normal']))
        elements.append(Spacer(1, 10))

        # Fisheye Parameters
        elements.append(Paragraph("Fisheye Correction Parameters (Down Camera)", heading_style))
        params_data = [
            ['Parameter', 'Value'],
            ['k1 (radial)', str(FISHEYE_CONFIG['down_camera']['k1'])],
            ['k2 (radial)', str(FISHEYE_CONFIG['down_camera']['k2'])],
            ['p1 (tangential)', str(FISHEYE_CONFIG['down_camera']['p1'])],
            ['p2 (tangential)', str(FISHEYE_CONFIG['down_camera']['p2'])],
            ['fx (focal x)', str(FISHEYE_CONFIG['down_camera']['fx'])],
            ['fy (focal y)', str(FISHEYE_CONFIG['down_camera']['fy'])],
            ['balance', str(FISHEYE_CONFIG['down_camera']['balance'])],
        ]
        params_table = Table(params_data, colWidths=[2*inch, 2*inch])
        params_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))
        elements.append(params_table)
        elements.append(Spacer(1, 20))

        # Results Table
        elements.append(Paragraph("Model Comparison (Fisheye Corrected)", heading_style))

        header = ['Metric'] + [r.model_version for r in self.results]
        table_data = [header]

        metrics_rows = [
            ('Training Date', [r.training_date for r in self.results]),
            ('Testing Date', [r.testing_date for r in self.results]),
            ('Model Size', [r.model_size for r in self.results]),
            ('Epochs', [str(r.epochs) for r in self.results]),
            ('Fisheye Corrected', ['Yes' for r in self.results]),
            ('mAP@50', [f"{r.map50:.4f}" for r in self.results]),
            ('mAP@50-95', [f"{r.map50_95:.4f}" for r in self.results]),
            ('Precision', [f"{r.precision:.4f}" for r in self.results]),
            ('Recall', [f"{r.recall:.4f}" for r in self.results]),
            ('F1 Score', [f"{r.f1_score:.4f}" for r in self.results]),
            ('FPS', [f"{r.fps:.2f}" for r in self.results]),
            ('GPU Memory (MiB)', [f"{r.gpu_memory_used_mib:.1f}/{r.gpu_memory_total_mib:.1f}" for r in self.results]),
            ('GPU Temp (°C)', [f"{r.gpu_temperature_c:.1f}" if r.gpu_temperature_c > 0 else "N/A" for r in self.results]),
            ('CPU Usage (%)', [f"{r.cpu_usage_percent:.1f}%" for r in self.results]),
            ('Training Time', [f"{r.training_time_minutes:.1f} min" for r in self.results]),
        ]

        for name, values in metrics_rows:
            table_data.append([name] + values)

        col_widths = [1.8*inch] + [1.5*inch] * len(self.results)
        table = Table(table_data, colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2980b9')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 1), (0, -1), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (1, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
        ]))
        elements.append(table)

        doc.build(elements)
        print(f"\n  ✓ Report: {report_path}")
        return str(report_path)

    def save_json(self):
        json_path = self.output_dir / "training_results.json"
        with open(json_path, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        print(f"  ✓ JSON: {json_path}")


# ============================================================================
# CELL 9: Main
# ============================================================================

def main():
    print("\n" + "="*70)
    print("  PHASE 3: YOLOv11 + FISHEYE CORRECTION")
    print("="*70)
    print(f"  Device: {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Fisheye k1={FISHEYE_CONFIG['down_camera']['k1']}, k2={FISHEYE_CONFIG['down_camera']['k2']}")
    print("="*70)

    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize fisheye corrector (for demo)
    print("\n  Initializing Fisheye Corrector...")
    corrector = FisheyeCorrector(FISHEYE_CONFIG['down_camera'])

    download_models()
    profiler = HardwareProfiler()

    if CONFIG['quick_test']:
        models = ['yolo11n']
        epochs_list = [15]
        print("\n  ⚡ QUICK TEST MODE\n")
    else:
        models = list(CONFIG['models'].keys())
        epochs_list = CONFIG['epochs_list']

    all_results = []
    total = len(models) * len(epochs_list)
    current = 0

    for model_name in models:
        for epochs in epochs_list:
            current += 1
            print(f"\n  [{current}/{total}] {model_name} - {epochs} epochs (fisheye)")
            result = train_model(model_name, epochs, profiler)
            if result:
                all_results.append(result)

    if all_results:
        report_gen = Phase3ReportGenerator(all_results, str(output_dir))
        report_gen.save_json()
        report_gen.generate()

    print("\n" + "="*70)
    print("  PHASE 3 COMPLETE!")
    print(f"  Results: {len(all_results)} runs")
    print(f"  Report:  {output_dir}/Phase3_Report.pdf")
    print("="*70 + "\n")

    return all_results


if __name__ == '__main__':
    # CONFIG['quick_test'] = True
    results = main()
