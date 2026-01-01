#!/usr/bin/env python3
"""
=============================================================================
PHASE 2: YOLOv11 STANDARD - PPE DETECTION RESEARCH
=============================================================================
Google Colab Script - Complete Pipeline

This script:
1. Downloads images from AWS S3 (optional)
2. Downloads YOLOv11 models (n/s/m/l/x)
3. Trains on Construction-PPE dataset (epochs: 15, 300, 600)
4. Collects comprehensive hardware metrics
5. Generates detailed PDF report with all curves

Run in Google Colab with GPU runtime.
=============================================================================
"""

# ============================================================================
# CELL 0: Install Dependencies
# ============================================================================
# !pip install ultralytics reportlab pandas matplotlib seaborn gputil psutil pillow boto3 openai -q

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
            print(f"  No files found in s3://{bucket_name}/{prefix}")
            return 0
        print(f"  Found {len(objects)} files")
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
    print("\n" + "="*70)
    print("  DOWNLOADING IMAGES FROM AWS S3")
    print("="*70)
    if os.environ.get('AWS_ACCESS_KEY_ID', '').startswith('YOUR_'):
        print("  ⚠ AWS credentials not set - using Ultralytics dataset")
        return
    for date in S3_CONFIG['dates_to_download']:
        for camera in S3_CONFIG['cameras']:
            prefix = f"{S3_CONFIG['image_prefix']}{date}/{camera}/"
            download_from_s3(S3_CONFIG['bucket_name'], prefix, f"s3_images/{date}/{camera}", S3_CONFIG['max_images_per_folder'])
    print("="*70 + "\n")

# ============================================================================
# CELL 3: Imports
# ============================================================================
import json
import time
import subprocess
import gc
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
import shutil

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image as PILImage
from ultralytics import YOLO

# Hardware monitoring
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    print("GPUtil not available, installing...")
    os.system("pip install gputil -q")
    import GPUtil
    HAS_GPUTIL = True

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("psutil not available, installing...")
    os.system("pip install psutil -q")
    import psutil
    HAS_PSUTIL = True

# ReportLab for PDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
    Image, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ============================================================================
# CELL 2: Configuration
# ============================================================================

CONFIG = {
    'phase': 2,
    'phase_name': 'YOLOv11 Standard',
    'dataset': 'construction-ppe.yaml',
    'imgsz': 640,
    'patience': 15,
    'workers': 4,
    'output_dir': 'ppe_research/phase2_yolov11',

    # Models to train
    'models': {
        'yolo11n': {'batch': 16, 'params': '2.6M', 'size': 'Nano'},
        'yolo11s': {'batch': 12, 'params': '9.4M', 'size': 'Small'},
        'yolo11m': {'batch': 8, 'params': '20.1M', 'size': 'Medium'},
        'yolo11l': {'batch': 4, 'params': '25.3M', 'size': 'Large'},
        'yolo11x': {'batch': 2, 'params': '56.9M', 'size': 'XLarge'},
    },

    # Epoch configurations
    'epochs_list': [15, 300, 600],

    # Quick test mode
    'quick_test': False,
}

# ============================================================================
# CELL 3: Comprehensive Hardware Metrics
# ============================================================================

@dataclass
class ComprehensiveMetrics:
    """Complete hardware and model metrics matching the required format"""
    # Model info
    model_name: str = ""
    model_version: str = ""
    model_size: str = ""
    params: str = ""
    epochs: int = 0

    # Dates
    training_date: str = ""
    testing_date: str = ""

    # Accuracy metrics
    map50: float = 0.0
    map50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Inference metrics
    fps: float = 0.0
    inference_latency_ms: float = 0.0

    # Memory metrics (in MiB for consistency with example)
    gpu_memory_used_mib: float = 0.0
    gpu_memory_total_mib: float = 0.0
    ram_used_mib: float = 0.0
    ram_total_mib: float = 0.0

    # Additional memory (simulated for edge devices)
    ddr_memory_used_mib: float = 0.0
    ddr_memory_total_mib: float = 0.0

    # Temperature and usage
    gpu_temperature_c: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_utilization_percent: float = 0.0

    # Power (if available)
    power_draw_w: float = 0.0

    # Training info
    training_time_minutes: float = 0.0

    # Paths to curves
    results_png: str = ""
    f1_curve: str = ""
    pr_curve: str = ""
    p_curve: str = ""
    r_curve: str = ""
    confusion_matrix: str = ""

    # Weights path
    weights_path: str = ""
    results_dir: str = ""


class HardwareProfiler:
    """Comprehensive hardware profiling"""

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gpu_name = torch.cuda.get_device_name(0) if self.device == 'cuda' else 'N/A'

    def get_all_metrics(self) -> Dict:
        """Get comprehensive hardware metrics"""
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

        # GPU metrics via PyTorch
        if self.device == 'cuda':
            metrics['gpu_memory_used_mib'] = torch.cuda.memory_allocated() / (1024**2)
            metrics['gpu_memory_total_mib'] = torch.cuda.get_device_properties(0).total_memory / (1024**2)

        # GPU metrics via GPUtil
        if HAS_GPUTIL:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    metrics['gpu_utilization_percent'] = gpu.load * 100
                    metrics['gpu_temperature_c'] = gpu.temperature
                    metrics['gpu_memory_used_mib'] = gpu.memoryUsed
                    metrics['gpu_memory_total_mib'] = gpu.memoryTotal
            except Exception as e:
                print(f"  Warning: GPUtil error: {e}")

        # CPU and RAM via psutil
        if HAS_PSUTIL:
            metrics['cpu_usage_percent'] = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            metrics['ram_used_mib'] = mem.used / (1024**2)
            metrics['ram_total_mib'] = mem.total / (1024**2)

        # Try nvidia-smi for power
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
        """Benchmark inference speed, return (fps, latency_ms)"""
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Warmup
        for _ in range(10):
            model.predict(dummy_img, verbose=False)

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        latencies = []
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            model.predict(dummy_img, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = np.mean(latencies)
        fps = 1000 / avg_latency

        return fps, avg_latency


# ============================================================================
# CELL 4: Download Models
# ============================================================================

def download_models():
    """Download all YOLOv11 base models"""
    print("\n" + "="*70)
    print("  DOWNLOADING YOLOv11 MODELS")
    print("="*70)

    for model_name in CONFIG['models'].keys():
        model_file = f"{model_name}.pt"
        if not Path(model_file).exists():
            print(f"  Downloading {model_file}...")
            model = YOLO(model_file)
            print(f"  ✓ {model_file} downloaded")
        else:
            print(f"  ✓ {model_file} already exists")

    print("="*70 + "\n")


# ============================================================================
# CELL 5: Training Function
# ============================================================================

def train_model(model_name: str, epochs: int, profiler: HardwareProfiler) -> Optional[ComprehensiveMetrics]:
    """Train a single model and collect all metrics"""

    model_config = CONFIG['models'][model_name]
    base_model = f"{model_name}.pt"

    # Version string (e.g., v11.0.n.e15)
    size_letter = model_name[-1]
    version = f"v11.0.{size_letter}.e{epochs}"

    run_name = f"{model_name}_e{epochs}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    output_dir = Path(CONFIG['output_dir']) / run_name

    print("\n" + "="*70)
    print(f"  TRAINING: {model_name.upper()} - {epochs} EPOCHS")
    print("="*70)
    print(f"  Version:    {version}")
    print(f"  Size:       {model_config['size']}")
    print(f"  Parameters: {model_config['params']}")
    print(f"  Batch Size: {model_config['batch']}")
    print("="*70)

    training_date = datetime.now().strftime("%b %d, %Y")
    start_time = time.time()

    try:
        # Load and train
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

        # Paths
        results_dir = Path(CONFIG['output_dir']) / run_name
        best_weights = results_dir / 'weights' / 'best.pt'

        # Load trained model
        trained_model = YOLO(str(best_weights))

        # Validate
        val_results = trained_model.val()

        # Hardware metrics
        hw = profiler.get_all_metrics()

        # Inference benchmark
        fps, latency = profiler.benchmark_inference(trained_model)

        # Calculate F1
        precision = float(val_results.box.mp)
        recall = float(val_results.box.mr)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Find curve paths
        def find_curve(pattern: str) -> str:
            matches = list(results_dir.glob(f"*{pattern}*"))
            return str(matches[0]) if matches else ""

        metrics = ComprehensiveMetrics(
            model_name=model_name,
            model_version=version,
            model_size=model_config['size'],
            params=model_config['params'],
            epochs=epochs,
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
            ddr_memory_used_mib=hw['gpu_memory_used_mib'],  # Approximate
            ddr_memory_total_mib=hw['gpu_memory_total_mib'],
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

        print(f"\n  ✓ Training Complete!")
        print(f"    mAP50:      {metrics.map50:.4f}")
        print(f"    mAP50-95:   {metrics.map50_95:.4f}")
        print(f"    F1 Score:   {metrics.f1_score:.4f}")
        print(f"    FPS:        {metrics.fps:.2f}")
        print(f"    GPU Temp:   {metrics.gpu_temperature_c:.1f}°C")
        print(f"    Time:       {training_time:.1f} min")

        # Cleanup
        del model, trained_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metrics

    except Exception as e:
        print(f"\n  ✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# CELL 6: Report Generator
# ============================================================================

class Phase2ReportGenerator:
    """Generate comprehensive PDF report matching required format"""

    def __init__(self, results: List[ComprehensiveMetrics], output_dir: str):
        self.results = results
        self.output_dir = Path(output_dir)

    def generate(self) -> str:
        """Generate the PDF report"""
        report_path = self.output_dir / "Phase2_Report.pdf"

        doc = SimpleDocTemplate(
            str(report_path),
            pagesize=landscape(A4),
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=20, alignment=TA_CENTER, spaceAfter=20)
        heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=14, spaceAfter=10)
        small_style = ParagraphStyle('Small', parent=styles['Normal'], fontSize=8)

        elements = []

        # Title Page
        elements.append(Paragraph("Phase 2: YOLOv11 Standard - PPE Detection Research", title_style))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}", styles['Normal']))
        elements.append(Spacer(1, 30))

        # Model Documentation Table (matching user's format)
        elements.append(Paragraph("Model Documentation", heading_style))

        # Group by model size
        models_by_size = {}
        for r in self.results:
            if r.model_size not in models_by_size:
                models_by_size[r.model_size] = []
            models_by_size[r.model_size].append(r)

        for size, size_results in models_by_size.items():
            elements.append(Paragraph(f"Model Size: {size}", styles['Heading3']))

            # Create table matching user's format
            header = ['Metric'] + [f"{r.model_version}" for r in size_results]
            table_data = [header]

            # Add all metrics rows
            metrics_rows = [
                ('Training Date', [r.training_date for r in size_results]),
                ('Testing Date', [r.testing_date for r in size_results]),
                ('Model Size', [r.model_size for r in size_results]),
                ('# of Epochs', [str(r.epochs) for r in size_results]),
                ('Model Version', [r.model_version for r in size_results]),
                ('mAP@50', [f"{r.map50:.4f}" for r in size_results]),
                ('mAP@50-95', [f"{r.map50_95:.4f}" for r in size_results]),
                ('Precision', [f"{r.precision:.4f}" for r in size_results]),
                ('Recall', [f"{r.recall:.4f}" for r in size_results]),
                ('F1 Score', [f"{r.f1_score:.4f}" for r in size_results]),
                ('FPS', [f"{r.fps:.2f}" for r in size_results]),
                ('GPU Memory (used/total)', [f"{r.gpu_memory_used_mib:.1f} / {r.gpu_memory_total_mib:.1f} MiB" for r in size_results]),
                ('RAM Usage (used/total)', [f"{r.ram_used_mib:.1f} / {r.ram_total_mib:.1f} MiB" for r in size_results]),
                ('GPU Temperature (°C)', [f"{r.gpu_temperature_c:.1f}°C" if r.gpu_temperature_c > 0 else "N/A" for r in size_results]),
                ('CPU Usage (%)', [f"{r.cpu_usage_percent:.1f}%" for r in size_results]),
                ('GPU Utilization (%)', [f"{r.gpu_utilization_percent:.1f}%" for r in size_results]),
                ('Power Draw (W)', [f"{r.power_draw_w:.1f}W" if r.power_draw_w > 0 else "N/A" for r in size_results]),
                ('Training Time', [f"{r.training_time_minutes:.1f} min" for r in size_results]),
            ]

            for metric_name, values in metrics_rows:
                table_data.append([metric_name] + values)

            # Create table
            col_widths = [2*inch] + [1.8*inch] * len(size_results)
            table = Table(table_data, colWidths=col_widths, repeatRows=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#34495e')),
                ('TEXTCOLOR', (0, 1), (0, -1), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 7),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (1, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            elements.append(table)
            elements.append(Spacer(1, 20))

        # Add curves section
        elements.append(PageBreak())
        elements.append(Paragraph("Training Curves", heading_style))

        for r in self.results:
            elements.append(Paragraph(f"{r.model_name} - {r.epochs} epochs ({r.model_version})", styles['Heading4']))

            # Add curves if available
            curves = [
                ('Results', r.results_png),
                ('F1 Curve', r.f1_curve),
                ('PR Curve', r.pr_curve),
                ('Confusion Matrix', r.confusion_matrix),
            ]

            curve_images = []
            for name, path in curves:
                if path and Path(path).exists():
                    try:
                        img = Image(path, width=3*inch, height=2.2*inch)
                        curve_images.append([Paragraph(name, small_style), img])
                    except:
                        pass

            if curve_images:
                # Create 2-column layout for curves
                while len(curve_images) < 4:
                    curve_images.append(['', ''])

                curve_table = Table([
                    [curve_images[0][0], curve_images[1][0]],
                    [curve_images[0][1] if len(curve_images[0]) > 1 else '', curve_images[1][1] if len(curve_images[1]) > 1 else ''],
                    [curve_images[2][0] if len(curve_images) > 2 else '', curve_images[3][0] if len(curve_images) > 3 else ''],
                    [curve_images[2][1] if len(curve_images) > 2 and len(curve_images[2]) > 1 else '',
                     curve_images[3][1] if len(curve_images) > 3 and len(curve_images[3]) > 1 else ''],
                ], colWidths=[4*inch, 4*inch])
                elements.append(curve_table)

            elements.append(Spacer(1, 15))

        # Build PDF
        doc.build(elements)
        print(f"\n  ✓ Report generated: {report_path}")
        return str(report_path)

    def save_results_json(self):
        """Save results to JSON"""
        json_path = self.output_dir / "training_results.json"
        with open(json_path, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        print(f"  ✓ Results saved: {json_path}")

    def generate_summary_chart(self):
        """Generate summary comparison chart"""
        if not self.results:
            return

        df = pd.DataFrame([asdict(r) for r in self.results])

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # mAP50 comparison
        pivot = df.pivot(index='model_name', columns='epochs', values='map50')
        pivot.plot(kind='bar', ax=axes[0, 0], colormap='viridis')
        axes[0, 0].set_title('mAP@50 by Model and Epochs')
        axes[0, 0].set_ylabel('mAP@50')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # F1 comparison
        pivot_f1 = df.pivot(index='model_name', columns='epochs', values='f1_score')
        pivot_f1.plot(kind='bar', ax=axes[0, 1], colormap='plasma')
        axes[0, 1].set_title('F1 Score by Model and Epochs')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # FPS vs Accuracy
        for epochs in df['epochs'].unique():
            subset = df[df['epochs'] == epochs]
            axes[1, 0].scatter(subset['fps'], subset['map50'], label=f'{epochs} epochs', s=100, alpha=0.7)
        axes[1, 0].set_xlabel('FPS')
        axes[1, 0].set_ylabel('mAP@50')
        axes[1, 0].set_title('Speed vs Accuracy Trade-off')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # GPU Temperature
        pivot_temp = df.pivot(index='model_name', columns='epochs', values='gpu_temperature_c')
        pivot_temp.plot(kind='bar', ax=axes[1, 1], colormap='coolwarm')
        axes[1, 1].set_title('GPU Temperature by Model')
        axes[1, 1].set_ylabel('Temperature (°C)')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'phase2_summary.png', dpi=150)
        plt.close()
        print(f"  ✓ Summary chart saved")


# ============================================================================
# CELL 7: Main Execution
# ============================================================================

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("  PHASE 2: YOLOv11 STANDARD - PPE DETECTION RESEARCH")
    print("="*70)
    print(f"  Dataset:    {CONFIG['dataset']}")
    print(f"  Image Size: {CONFIG['imgsz']}")
    print(f"  Device:     {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
    print("="*70)

    # Setup
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    download_models()
    profiler = HardwareProfiler()

    # Determine training scope
    if CONFIG['quick_test']:
        models_to_train = ['yolo11n']
        epochs_to_train = [15]
        print("\n  ⚡ QUICK TEST MODE\n")
    else:
        models_to_train = list(CONFIG['models'].keys())
        epochs_to_train = CONFIG['epochs_list']

    # Train all
    all_results: List[ComprehensiveMetrics] = []
    total = len(models_to_train) * len(epochs_to_train)
    current = 0

    for model_name in models_to_train:
        for epochs in epochs_to_train:
            current += 1
            print(f"\n  [{current}/{total}] {model_name} - {epochs} epochs")
            result = train_model(model_name, epochs, profiler)
            if result:
                all_results.append(result)

    # Generate reports
    if all_results:
        report_gen = Phase2ReportGenerator(all_results, str(output_dir))
        report_gen.save_results_json()
        report_gen.generate_summary_chart()
        report_gen.generate()

    print("\n" + "="*70)
    print("  PHASE 2 COMPLETE!")
    print(f"  Results: {len(all_results)} runs")
    print(f"  Report:  {output_dir}/Phase2_Report.pdf")
    print("="*70 + "\n")

    return all_results


if __name__ == '__main__':
    # CONFIG['quick_test'] = True  # Uncomment for quick test
    results = main()
