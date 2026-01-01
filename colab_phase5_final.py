#!/usr/bin/env python3
"""
=============================================================================
PHASE 5: FINAL OPTIMIZATION & MODEL EXPORT - PPE DETECTION RESEARCH
=============================================================================
Google Colab Script - Complete Pipeline

This script:
1. Downloads images from AWS S3 (optional)
2. Takes best model configuration from Phase 4
3. Trains with extended epochs (600-1000)
4. Exports to multiple formats (ONNX, TensorRT)
5. Generates final comprehensive research report

Run in Google Colab with GPU runtime.
=============================================================================
"""

# ============================================================================
# CELL 0: Install Dependencies
# ============================================================================
# !pip install ultralytics reportlab pandas matplotlib seaborn gputil psutil onnx onnxruntime boto3 openai -q

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
import gc
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import subprocess
import shutil

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

try:
    import onnx
    import onnxruntime
    HAS_ONNX = True
except:
    os.system("pip install onnx onnxruntime -q")
    HAS_ONNX = True

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak, KeepTogether
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ============================================================================
# CELL 2: Configuration
# ============================================================================

CONFIG = {
    'phase': 5,
    'phase_name': 'Final Optimization & Export',
    'dataset': 'construction-ppe.yaml',
    'imgsz': 640,
    'workers': 4,
    'output_dir': 'ppe_research/phase5_final',

    # Best model from Phase 4 (update based on your results)
    'best_model': 'yolo11l',
    'model_config': {'batch': 4, 'params': '25.3M', 'size': 'Large'},

    # Best hyperparameters from Phase 4
    'best_hyperparams': {
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'mosaic': 1.0,
        'mixup': 0.0,
    },

    # Extended training
    'epochs_list': [600, 1000],

    # Export formats
    'export_formats': ['onnx', 'torchscript'],

    'quick_test': False,
}

# ============================================================================
# CELL 3: Comprehensive Metrics
# ============================================================================

@dataclass
class FinalMetrics:
    """Complete metrics for final model"""
    model_name: str = ""
    model_version: str = ""
    model_size: str = ""
    params: str = ""
    epochs: int = 0

    # Best hyperparameters
    lr0: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005

    training_date: str = ""
    testing_date: str = ""

    # Accuracy
    map50: float = 0.0
    map50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Per-class AP
    class_aps: Dict = None

    # Inference (PyTorch)
    fps_pytorch: float = 0.0
    latency_pytorch_ms: float = 0.0

    # Inference (ONNX)
    fps_onnx: float = 0.0
    latency_onnx_ms: float = 0.0

    # Hardware
    gpu_memory_used_mib: float = 0.0
    gpu_memory_total_mib: float = 0.0
    gpu_temperature_c: float = 0.0
    cpu_usage_percent: float = 0.0
    power_draw_w: float = 0.0

    training_time_hours: float = 0.0

    # File sizes
    pytorch_size_mb: float = 0.0
    onnx_size_mb: float = 0.0

    # Paths
    pytorch_path: str = ""
    onnx_path: str = ""
    results_dir: str = ""

    # Curves
    results_png: str = ""
    f1_curve: str = ""
    pr_curve: str = ""
    confusion_matrix: str = ""


# ============================================================================
# CELL 4: Hardware Profiler
# ============================================================================

class HardwareProfiler:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_metrics(self) -> Dict:
        metrics = {
            'gpu_memory_used_mib': 0.0,
            'gpu_memory_total_mib': 0.0,
            'gpu_temperature_c': 0.0,
            'cpu_usage_percent': psutil.cpu_percent(interval=0.1),
            'power_draw_w': 0.0,
        }

        if self.device == 'cuda':
            metrics['gpu_memory_used_mib'] = torch.cuda.memory_allocated() / (1024**2)
            metrics['gpu_memory_total_mib'] = torch.cuda.get_device_properties(0).total_memory / (1024**2)

        if HAS_GPUTIL:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    metrics['gpu_temperature_c'] = gpus[0].temperature
            except:
                pass

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

    def benchmark_pytorch(self, model, runs=100) -> Tuple[float, float]:
        dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(10):
            model.predict(dummy, verbose=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        latencies = []
        for _ in range(runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            model.predict(dummy, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

        return 1000 / np.mean(latencies), np.mean(latencies)

    def benchmark_onnx(self, onnx_path: str, runs=100) -> Tuple[float, float]:
        """Benchmark ONNX model"""
        try:
            import onnxruntime as ort

            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(onnx_path, providers=providers)

            input_name = session.get_inputs()[0].name
            dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)

            # Warmup
            for _ in range(10):
                session.run(None, {input_name: dummy})

            latencies = []
            for _ in range(runs):
                start = time.perf_counter()
                session.run(None, {input_name: dummy})
                latencies.append((time.perf_counter() - start) * 1000)

            return 1000 / np.mean(latencies), np.mean(latencies)

        except Exception as e:
            print(f"  ONNX benchmark failed: {e}")
            return 0.0, 0.0


# ============================================================================
# CELL 5: Training and Export
# ============================================================================

def train_final_model(epochs: int, profiler: HardwareProfiler) -> Optional[FinalMetrics]:
    """Train the final optimized model"""

    model_name = CONFIG['best_model']
    hp = CONFIG['best_hyperparams']

    run_name = f"final_{model_name}_e{epochs}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    version = f"v11.0.final.e{epochs}"

    print("\n" + "="*70)
    print(f"  FINAL TRAINING: {model_name.upper()} - {epochs} EPOCHS")
    print("="*70)
    print(f"  Version:      {version}")
    print(f"  lr0={hp['lr0']}, momentum={hp['momentum']}")
    print(f"  weight_decay={hp['weight_decay']}, mosaic={hp['mosaic']}")
    print("="*70)

    training_date = datetime.now().strftime("%b %d, %Y")
    start_time = time.time()

    try:
        model = YOLO(f"{model_name}.pt")

        results = model.train(
            data=CONFIG['dataset'],
            epochs=epochs,
            imgsz=CONFIG['imgsz'],
            batch=CONFIG['model_config']['batch'],
            patience=50,  # Higher patience for final training
            workers=CONFIG['workers'],
            project=CONFIG['output_dir'],
            name=run_name,
            exist_ok=True,
            plots=True,
            save=True,
            verbose=True,
            amp=True,
            # Best hyperparameters
            lr0=hp['lr0'],
            lrf=hp['lrf'],
            momentum=hp['momentum'],
            weight_decay=hp['weight_decay'],
            warmup_epochs=hp['warmup_epochs'],
            mosaic=hp['mosaic'],
            mixup=hp['mixup'],
        )

        training_time = (time.time() - start_time) / 3600  # hours
        testing_date = datetime.now().strftime("%b %d, %Y")

        results_dir = Path(CONFIG['output_dir']) / run_name
        best_weights = results_dir / 'weights' / 'best.pt'

        # Load trained model
        trained_model = YOLO(str(best_weights))

        # Validate
        print("\n  Validating model...")
        val_results = trained_model.val()

        # Get per-class AP
        class_aps = {}
        if hasattr(val_results.box, 'ap_class_index'):
            names = trained_model.names
            for i, ap in enumerate(val_results.box.ap50):
                if i < len(names):
                    class_aps[names[i]] = float(ap)

        # Hardware metrics
        hw = profiler.get_metrics()

        # PyTorch benchmark
        print("  Benchmarking PyTorch...")
        fps_pt, lat_pt = profiler.benchmark_pytorch(trained_model)

        # Export to ONNX
        print("  Exporting to ONNX...")
        onnx_path = results_dir / f"final_{model_name}_e{epochs}.onnx"
        trained_model.export(format='onnx', imgsz=640, simplify=True)

        # Find exported ONNX file
        exported_onnx = list(results_dir.glob("**/*.onnx"))
        if exported_onnx:
            shutil.copy(exported_onnx[0], onnx_path)
            onnx_size = onnx_path.stat().st_size / (1024**2)

            # ONNX benchmark
            print("  Benchmarking ONNX...")
            fps_onnx, lat_onnx = profiler.benchmark_onnx(str(onnx_path))
        else:
            onnx_size = 0.0
            fps_onnx, lat_onnx = 0.0, 0.0

        # Calculate metrics
        precision = float(val_results.box.mp)
        recall = float(val_results.box.mr)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        def find_curve(pattern: str) -> str:
            matches = list(results_dir.glob(f"*{pattern}*"))
            return str(matches[0]) if matches else ""

        pytorch_size = best_weights.stat().st_size / (1024**2)

        metrics = FinalMetrics(
            model_name=model_name,
            model_version=version,
            model_size=CONFIG['model_config']['size'],
            params=CONFIG['model_config']['params'],
            epochs=epochs,
            lr0=hp['lr0'],
            momentum=hp['momentum'],
            weight_decay=hp['weight_decay'],
            training_date=training_date,
            testing_date=testing_date,
            map50=float(val_results.box.map50),
            map50_95=float(val_results.box.map),
            precision=precision,
            recall=recall,
            f1_score=f1,
            class_aps=class_aps,
            fps_pytorch=fps_pt,
            latency_pytorch_ms=lat_pt,
            fps_onnx=fps_onnx,
            latency_onnx_ms=lat_onnx,
            gpu_memory_used_mib=hw['gpu_memory_used_mib'],
            gpu_memory_total_mib=hw['gpu_memory_total_mib'],
            gpu_temperature_c=hw['gpu_temperature_c'],
            cpu_usage_percent=hw['cpu_usage_percent'],
            power_draw_w=hw['power_draw_w'],
            training_time_hours=training_time,
            pytorch_size_mb=pytorch_size,
            onnx_size_mb=onnx_size,
            pytorch_path=str(best_weights),
            onnx_path=str(onnx_path) if onnx_path.exists() else "",
            results_dir=str(results_dir),
            results_png=find_curve('results'),
            f1_curve=find_curve('F1_curve'),
            pr_curve=find_curve('PR_curve'),
            confusion_matrix=find_curve('confusion_matrix.png'),
        )

        print(f"\n  ✓ Training Complete!")
        print(f"    mAP50:          {metrics.map50:.4f}")
        print(f"    mAP50-95:       {metrics.map50_95:.4f}")
        print(f"    F1 Score:       {metrics.f1_score:.4f}")
        print(f"    FPS (PyTorch):  {metrics.fps_pytorch:.2f}")
        print(f"    FPS (ONNX):     {metrics.fps_onnx:.2f}")
        print(f"    Training Time:  {training_time:.2f} hours")

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
# CELL 6: Final Report Generator
# ============================================================================

class FinalReportGenerator:
    def __init__(self, results: List[FinalMetrics], output_dir: str):
        self.results = results
        self.output_dir = Path(output_dir)

    def generate(self) -> str:
        report_path = self.output_dir / "Phase5_Final_Report.pdf"

        doc = SimpleDocTemplate(
            str(report_path),
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=22, alignment=TA_CENTER, spaceAfter=20)
        heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=14, spaceAfter=10)
        subheading_style = ParagraphStyle('SubHeading', parent=styles['Heading3'], fontSize=11, spaceAfter=8)

        elements = []

        # Title Page
        elements.append(Paragraph("PPE Detection Research", title_style))
        elements.append(Paragraph("Phase 5: Final Model Report", styles['Heading2']))
        elements.append(Spacer(1, 10))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}", styles['Normal']))
        elements.append(Spacer(1, 30))

        if not self.results:
            elements.append(Paragraph("No results available.", styles['Normal']))
            doc.build(elements)
            return str(report_path)

        # Get best result
        best = max(self.results, key=lambda x: x.map50)

        # Executive Summary
        elements.append(Paragraph("Executive Summary", heading_style))
        summary_text = f"""
        The final PPE detection model achieves <b>mAP@50 of {best.map50:.4f}</b> and <b>mAP@50-95 of {best.map50_95:.4f}</b>
        after {best.epochs} epochs of training. The model runs at <b>{best.fps_pytorch:.1f} FPS</b> (PyTorch) and
        <b>{best.fps_onnx:.1f} FPS</b> (ONNX) on the test hardware. Training took {best.training_time_hours:.1f} hours.
        """
        elements.append(Paragraph(summary_text, styles['Normal']))
        elements.append(Spacer(1, 20))

        # Model Documentation Table
        elements.append(Paragraph("Model Documentation", heading_style))

        model_doc = [
            ['Metric', 'Value'],
            ['Model Name', best.model_name],
            ['Model Version', best.model_version],
            ['Model Size', best.model_size],
            ['Parameters', best.params],
            ['Training Date', best.training_date],
            ['Testing Date', best.testing_date],
            ['# of Epochs', str(best.epochs)],
        ]

        model_table = Table(model_doc, colWidths=[2.5*inch, 3*inch])
        model_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
        ]))
        elements.append(model_table)
        elements.append(Spacer(1, 20))

        # Accuracy Metrics
        elements.append(Paragraph("Accuracy Metrics", heading_style))

        accuracy_data = [
            ['Metric', 'Value'],
            ['mAP@50', f"{best.map50:.4f}"],
            ['mAP@50-95', f"{best.map50_95:.4f}"],
            ['Precision', f"{best.precision:.4f}"],
            ['Recall', f"{best.recall:.4f}"],
            ['F1 Score', f"{best.f1_score:.4f}"],
        ]

        acc_table = Table(accuracy_data, colWidths=[2.5*inch, 3*inch])
        acc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
        ]))
        elements.append(acc_table)
        elements.append(Spacer(1, 20))

        # Per-Class Performance
        if best.class_aps:
            elements.append(Paragraph("Per-Class Performance (AP@50)", subheading_style))
            class_data = [['Class', 'AP@50']]
            for cls, ap in sorted(best.class_aps.items(), key=lambda x: x[1], reverse=True):
                class_data.append([cls, f"{ap:.4f}"])

            class_table = Table(class_data, colWidths=[2.5*inch, 2*inch])
            class_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
            ]))
            elements.append(class_table)
            elements.append(Spacer(1, 20))

        # Inference Performance
        elements.append(Paragraph("Inference Performance", heading_style))

        inference_data = [
            ['Metric', 'PyTorch', 'ONNX'],
            ['FPS', f"{best.fps_pytorch:.2f}", f"{best.fps_onnx:.2f}" if best.fps_onnx > 0 else "N/A"],
            ['Latency (ms)', f"{best.latency_pytorch_ms:.2f}", f"{best.latency_onnx_ms:.2f}" if best.latency_onnx_ms > 0 else "N/A"],
            ['Model Size (MB)', f"{best.pytorch_size_mb:.1f}", f"{best.onnx_size_mb:.1f}" if best.onnx_size_mb > 0 else "N/A"],
        ]

        inf_table = Table(inference_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        inf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9b59b6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
        ]))
        elements.append(inf_table)
        elements.append(Spacer(1, 20))

        # Hardware Metrics
        elements.append(Paragraph("Hardware Metrics", heading_style))

        hw_data = [
            ['Metric', 'Value'],
            ['GPU Memory (used/total)', f"{best.gpu_memory_used_mib:.1f} / {best.gpu_memory_total_mib:.1f} MiB"],
            ['GPU Temperature', f"{best.gpu_temperature_c:.1f}°C" if best.gpu_temperature_c > 0 else "N/A"],
            ['CPU Usage', f"{best.cpu_usage_percent:.1f}%"],
            ['Power Draw', f"{best.power_draw_w:.1f}W" if best.power_draw_w > 0 else "N/A"],
            ['Training Time', f"{best.training_time_hours:.2f} hours"],
        ]

        hw_table = Table(hw_data, colWidths=[2.5*inch, 3*inch])
        hw_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
        ]))
        elements.append(hw_table)
        elements.append(Spacer(1, 20))

        # Hyperparameters Used
        elements.append(Paragraph("Optimized Hyperparameters", heading_style))

        hp_data = [
            ['Parameter', 'Value'],
            ['Learning Rate (lr0)', str(best.lr0)],
            ['Momentum', str(best.momentum)],
            ['Weight Decay', str(best.weight_decay)],
        ]

        hp_table = Table(hp_data, colWidths=[2.5*inch, 3*inch])
        hp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f39c12')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
        ]))
        elements.append(hp_table)
        elements.append(Spacer(1, 20))

        # Export Paths
        elements.append(Paragraph("Model Files", heading_style))

        export_data = [
            ['Format', 'Path', 'Size'],
            ['PyTorch (.pt)', Path(best.pytorch_path).name, f"{best.pytorch_size_mb:.1f} MB"],
            ['ONNX (.onnx)', Path(best.onnx_path).name if best.onnx_path else "N/A", f"{best.onnx_size_mb:.1f} MB" if best.onnx_size_mb > 0 else "N/A"],
        ]

        export_table = Table(export_data, colWidths=[1.5*inch, 3*inch, 1*inch])
        export_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1abc9c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))
        elements.append(export_table)

        # Build PDF
        doc.build(elements)
        print(f"\n  ✓ Final Report: {report_path}")
        return str(report_path)

    def save_json(self):
        json_path = self.output_dir / "final_results.json"
        results_data = []
        for r in self.results:
            data = asdict(r)
            # Convert class_aps dict to serializable format
            if data['class_aps'] is None:
                data['class_aps'] = {}
            results_data.append(data)

        with open(json_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"  ✓ JSON: {json_path}")


# ============================================================================
# CELL 7: Main
# ============================================================================

def main():
    print("\n" + "="*70)
    print("  PHASE 5: FINAL OPTIMIZATION & EXPORT")
    print("="*70)
    print(f"  Device: {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Model:  {CONFIG['best_model']}")
    print(f"  Epochs: {CONFIG['epochs_list']}")
    print("="*70)

    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    profiler = HardwareProfiler()

    # Download model if needed
    model_file = f"{CONFIG['best_model']}.pt"
    if not Path(model_file).exists():
        print(f"\n  Downloading {model_file}...")
        YOLO(model_file)

    if CONFIG['quick_test']:
        epochs_list = [100]  # Quick test with 100 epochs
        print("\n  ⚡ QUICK TEST MODE (100 epochs)\n")
    else:
        epochs_list = CONFIG['epochs_list']

    all_results = []

    for epochs in epochs_list:
        result = train_final_model(epochs, profiler)
        if result:
            all_results.append(result)

    if all_results:
        report_gen = FinalReportGenerator(all_results, str(output_dir))
        report_gen.save_json()
        report_gen.generate()

        # Print final summary
        best = max(all_results, key=lambda x: x.map50)
        print("\n" + "="*70)
        print("  FINAL MODEL SUMMARY")
        print("="*70)
        print(f"  Model:        {best.model_name} ({best.epochs} epochs)")
        print(f"  mAP@50:       {best.map50:.4f}")
        print(f"  mAP@50-95:    {best.map50_95:.4f}")
        print(f"  F1 Score:     {best.f1_score:.4f}")
        print(f"  FPS:          {best.fps_pytorch:.2f} (PyTorch)")
        print(f"  Model Size:   {best.pytorch_size_mb:.1f} MB")
        print("="*70)
        print(f"\n  Files:")
        print(f"    PyTorch: {best.pytorch_path}")
        if best.onnx_path:
            print(f"    ONNX:    {best.onnx_path}")
        print(f"    Report:  {output_dir}/Phase5_Final_Report.pdf")
        print("="*70 + "\n")

    print("\n  PHASE 5 COMPLETE!")
    print(f"  PPE DETECTION RESEARCH FINISHED!\n")

    return all_results


if __name__ == '__main__':
    # CONFIG['quick_test'] = True
    results = main()
