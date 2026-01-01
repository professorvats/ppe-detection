#!/usr/bin/env python3
"""
=============================================================================
PHASE 1: YOLOv8 BASELINE - PPE DETECTION RESEARCH
=============================================================================
Google Colab Script - Complete Pipeline

This script:
1. Downloads images from AWS S3 (optional)
2. Downloads YOLOv8 models (n/s/m/l/x)
3. Trains on Construction-PPE dataset (epochs: 15, 300, 600)
4. Collects hardware metrics (FPS, memory, temperature)
5. Generates comprehensive PDF report

Run in Google Colab with GPU runtime.
=============================================================================
"""

# ============================================================================
# CELL 0: Install Dependencies
# ============================================================================
# !pip install ultralytics reportlab pandas matplotlib seaborn gputil psutil boto3 openai -q

# ============================================================================
# CELL 1: AWS & OpenAI Configuration (SET YOUR CREDENTIALS)
# ============================================================================
import os

# AWS Credentials - Set these before running
os.environ['AWS_ACCESS_KEY_ID'] = 'AKIA3DIORHCHHBJTPOOC'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'YOUR_AWS_SECRET_ACCESS_KEY'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

# OpenAI API Key (optional - for enhanced report analysis)
os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_API_KEY'

# S3 Bucket Configuration
S3_CONFIG = {
    'bucket_name': 'your-ppe-bucket',
    'image_prefix': 'images/',
    'dates_to_download': ['10-31', '11-7'],
    'cameras': ['down'],
    'max_images_per_folder': None,  # None = download all
}

# ============================================================================
# CELL 2: AWS S3 Download Functions
# ============================================================================
def download_from_s3(bucket_name, prefix, local_dir, max_files=None):
    """Download files from S3 bucket"""
    try:
        import boto3
        from pathlib import Path
        from concurrent.futures import ThreadPoolExecutor

        s3 = boto3.client('s3')
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        # List objects
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=max_files or 1000)
        objects = response.get('Contents', [])

        if not objects:
            print(f"  No files found in s3://{bucket_name}/{prefix}")
            return 0

        print(f"  Found {len(objects)} files in s3://{bucket_name}/{prefix}")

        downloaded = 0
        for obj in objects:
            key = obj['Key']
            filename = os.path.basename(key)
            if filename:
                local_path = os.path.join(local_dir, filename)
                s3.download_file(bucket_name, key, local_path)
                downloaded += 1
                if downloaded % 50 == 0:
                    print(f"    Downloaded {downloaded}/{len(objects)}")

        print(f"  ✓ Downloaded {downloaded} files to {local_dir}")
        return downloaded

    except Exception as e:
        print(f"  S3 download error: {e}")
        return 0


def download_ppe_images():
    """Download PPE images from configured S3 bucket"""
    print("\n" + "="*70)
    print("  DOWNLOADING IMAGES FROM AWS S3")
    print("="*70)

    if os.environ.get('AWS_ACCESS_KEY_ID', '').startswith('YOUR_'):
        print("  ⚠ AWS credentials not configured - skipping S3 download")
        print("  Using Ultralytics Construction-PPE dataset instead")
        return

    for date in S3_CONFIG['dates_to_download']:
        for camera in S3_CONFIG['cameras']:
            prefix = f"{S3_CONFIG['image_prefix']}{date}/{camera}/"
            local_dir = f"s3_images/{date}/{camera}"
            download_from_s3(
                S3_CONFIG['bucket_name'],
                prefix,
                local_dir,
                S3_CONFIG['max_images_per_folder']
            )

    print("="*70 + "\n")


# ============================================================================
# CELL 3: Setup and Imports
# ============================================================================

import os
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import shutil

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

# Optional imports for hardware monitoring
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ReportLab for PDF generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ============================================================================
# CELL 2: Configuration
# ============================================================================

CONFIG = {
    'phase': 1,
    'phase_name': 'YOLOv8 Baseline',
    'dataset': 'construction-ppe.yaml',
    'imgsz': 640,
    'patience': 15,
    'workers': 4,
    'output_dir': 'ppe_research/phase1_yolov8',

    # Models to train
    'models': {
        'yolov8n': {'batch': 16, 'params': '3.2M'},
        'yolov8s': {'batch': 12, 'params': '11.2M'},
        'yolov8m': {'batch': 8, 'params': '25.9M'},
        'yolov8l': {'batch': 4, 'params': '43.7M'},
        'yolov8x': {'batch': 2, 'params': '68.2M'},
    },

    # Epoch configurations
    'epochs_list': [15, 300, 600],

    # For quick testing, set to True
    'quick_test': False,  # Set True to only run nano with 15 epochs
}

# ============================================================================
# CELL 3: Hardware Profiler
# ============================================================================

@dataclass
class HardwareMetrics:
    """Container for hardware metrics"""
    gpu_name: str = ""
    gpu_memory_total_gb: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_utilization_percent: float = 0.0
    gpu_temperature_c: float = 0.0
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    inference_fps: float = 0.0
    inference_latency_ms: float = 0.0


class HardwareProfiler:
    """Collect hardware metrics during training and inference"""

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gpu_name = torch.cuda.get_device_name(0) if self.device == 'cuda' else 'N/A'

    def get_gpu_metrics(self) -> Dict:
        """Get current GPU metrics"""
        metrics = {
            'gpu_name': self.gpu_name,
            'gpu_memory_total_gb': 0.0,
            'gpu_memory_used_gb': 0.0,
            'gpu_utilization_percent': 0.0,
            'gpu_temperature_c': 0.0,
        }

        if self.device == 'cuda':
            # PyTorch memory
            metrics['gpu_memory_used_gb'] = torch.cuda.memory_allocated() / 1e9
            metrics['gpu_memory_total_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9

            # nvidia-smi metrics
            if HAS_GPUTIL:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        metrics['gpu_utilization_percent'] = gpu.load * 100
                        metrics['gpu_temperature_c'] = gpu.temperature
                except:
                    pass

        return metrics

    def get_cpu_metrics(self) -> Dict:
        """Get CPU and RAM metrics"""
        metrics = {'cpu_percent': 0.0, 'ram_percent': 0.0}
        if HAS_PSUTIL:
            metrics['cpu_percent'] = psutil.cpu_percent()
            metrics['ram_percent'] = psutil.virtual_memory().percent
        return metrics

    def benchmark_inference(self, model, num_images: int = 100) -> Dict:
        """Benchmark model inference speed"""
        # Create dummy images
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Warmup
        for _ in range(10):
            model.predict(dummy_img, verbose=False)

        # Benchmark
        latencies = []
        for _ in range(num_images):
            start = time.perf_counter()
            model.predict(dummy_img, verbose=False)
            latencies.append((time.perf_counter() - start) * 1000)

        return {
            'inference_fps': 1000 / np.mean(latencies),
            'inference_latency_ms': np.mean(latencies),
            'latency_std_ms': np.std(latencies),
        }

    def collect_all_metrics(self, model=None) -> HardwareMetrics:
        """Collect all hardware metrics"""
        gpu = self.get_gpu_metrics()
        cpu = self.get_cpu_metrics()
        inference = self.benchmark_inference(model) if model else {'inference_fps': 0, 'inference_latency_ms': 0}

        return HardwareMetrics(
            gpu_name=gpu['gpu_name'],
            gpu_memory_total_gb=gpu['gpu_memory_total_gb'],
            gpu_memory_used_gb=gpu['gpu_memory_used_gb'],
            gpu_utilization_percent=gpu['gpu_utilization_percent'],
            gpu_temperature_c=gpu['gpu_temperature_c'],
            cpu_percent=cpu['cpu_percent'],
            ram_percent=cpu['ram_percent'],
            inference_fps=inference['inference_fps'],
            inference_latency_ms=inference['inference_latency_ms'],
        )

# ============================================================================
# CELL 4: Training Result Container
# ============================================================================

@dataclass
class TrainingResult:
    """Container for a single training run"""
    phase: int
    model_name: str
    model_version: str  # e.g., "v8.0.n.e15"
    params: str
    epochs: int

    # Dates
    training_date: str
    testing_date: str

    # Final metrics
    map50: float
    map50_95: float
    precision: float
    recall: float
    f1_score: float

    # Hardware metrics
    fps: float
    gpu_memory_gb: float
    gpu_temperature_c: float
    cpu_percent: float

    # Paths
    weights_path: str
    results_dir: str

    # Training time
    training_time_minutes: float


# ============================================================================
# CELL 5: Model Downloader
# ============================================================================

def download_models():
    """Download all YOLOv8 base models"""
    print("\n" + "="*70)
    print("  DOWNLOADING YOLOv8 MODELS")
    print("="*70)

    models_to_download = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']

    for model_name in models_to_download:
        if not Path(model_name).exists():
            print(f"  Downloading {model_name}...")
            model = YOLO(model_name)
            print(f"  ✓ {model_name} downloaded")
        else:
            print(f"  ✓ {model_name} already exists")

    print("="*70 + "\n")

# ============================================================================
# CELL 6: Training Function
# ============================================================================

def train_model(model_name: str, epochs: int, config: dict, profiler: HardwareProfiler) -> Optional[TrainingResult]:
    """Train a single model configuration"""

    model_config = config['models'][model_name]
    base_model = f"{model_name}.pt"

    # Create version string (e.g., v8.0.n.e15)
    size_letter = model_name[-1]  # n, s, m, l, x
    version = f"v8.0.{size_letter}.e{epochs}"

    run_name = f"{model_name}_e{epochs}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    output_dir = Path(config['output_dir']) / run_name

    print("\n" + "="*70)
    print(f"  TRAINING: {model_name.upper()} - {epochs} EPOCHS")
    print("="*70)
    print(f"  Version:    {version}")
    print(f"  Parameters: {model_config['params']}")
    print(f"  Batch Size: {model_config['batch']}")
    print(f"  Output:     {output_dir}")
    print("="*70)

    training_date = datetime.now().strftime("%b %d, %Y")
    start_time = time.time()

    try:
        # Load model
        model = YOLO(base_model)

        # Train
        results = model.train(
            data=config['dataset'],
            epochs=epochs,
            imgsz=config['imgsz'],
            batch=model_config['batch'],
            patience=config['patience'],
            workers=config['workers'],
            project=config['output_dir'],
            name=run_name,
            exist_ok=True,
            plots=True,
            save=True,
            verbose=True,
            amp=True,
        )

        training_time = (time.time() - start_time) / 60  # minutes
        testing_date = datetime.now().strftime("%b %d, %Y")

        # Get best weights path
        best_weights = Path(config['output_dir']) / run_name / 'weights' / 'best.pt'

        # Load trained model for validation and benchmarking
        trained_model = YOLO(str(best_weights))

        # Validate
        val_results = trained_model.val()

        # Collect hardware metrics
        hw_metrics = profiler.collect_all_metrics(trained_model)

        # Calculate F1 score
        precision = float(val_results.box.mp)
        recall = float(val_results.box.mr)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        result = TrainingResult(
            phase=config['phase'],
            model_name=model_name,
            model_version=version,
            params=model_config['params'],
            epochs=epochs,
            training_date=training_date,
            testing_date=testing_date,
            map50=float(val_results.box.map50),
            map50_95=float(val_results.box.map),
            precision=precision,
            recall=recall,
            f1_score=f1,
            fps=hw_metrics.inference_fps,
            gpu_memory_gb=hw_metrics.gpu_memory_used_gb,
            gpu_temperature_c=hw_metrics.gpu_temperature_c,
            cpu_percent=hw_metrics.cpu_percent,
            weights_path=str(best_weights),
            results_dir=str(output_dir),
            training_time_minutes=training_time,
        )

        print(f"\n  ✓ Training Complete!")
        print(f"    mAP50:     {result.map50:.4f}")
        print(f"    mAP50-95:  {result.map50_95:.4f}")
        print(f"    F1 Score:  {result.f1_score:.4f}")
        print(f"    FPS:       {result.fps:.1f}")
        print(f"    Time:      {training_time:.1f} min")

        return result

    except Exception as e:
        print(f"\n  ✗ Training failed: {e}")
        return None

# ============================================================================
# CELL 7: Report Generator
# ============================================================================

class PhaseReportGenerator:
    """Generate PDF report for a training phase"""

    def __init__(self, phase: int, phase_name: str, results: List[TrainingResult], output_dir: str):
        self.phase = phase
        self.phase_name = phase_name
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self) -> str:
        """Generate the PDF report"""
        report_path = self.output_dir / f"Phase{self.phase}_Report.pdf"

        doc = SimpleDocTemplate(
            str(report_path),
            pagesize=A4,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            alignment=TA_CENTER,
            spaceAfter=30,
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
        )

        elements = []

        # Title
        elements.append(Paragraph(f"Phase {self.phase}: {self.phase_name}", title_style))
        elements.append(Paragraph("PPE Detection Research Report", styles['Heading2']))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}", styles['Normal']))
        elements.append(Spacer(1, 30))

        # Summary Table
        elements.append(Paragraph("Model Comparison Summary", heading_style))

        # Create comparison table
        table_data = [
            ['Model', 'Version', 'Epochs', 'mAP50', 'mAP50-95', 'F1', 'FPS', 'Memory', 'Temp']
        ]

        for r in self.results:
            table_data.append([
                r.model_name,
                r.model_version,
                str(r.epochs),
                f"{r.map50:.3f}",
                f"{r.map50_95:.3f}",
                f"{r.f1_score:.3f}",
                f"{r.fps:.1f}",
                f"{r.gpu_memory_gb:.1f}GB",
                f"{r.gpu_temperature_c:.0f}°C" if r.gpu_temperature_c > 0 else "N/A",
            ])

        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 30))

        # Detailed Results per Model
        elements.append(Paragraph("Detailed Results by Model", heading_style))

        for r in self.results:
            elements.append(Paragraph(f"{r.model_name} - {r.epochs} Epochs", styles['Heading3']))

            detail_data = [
                ['Metric', 'Value'],
                ['Model Version', r.model_version],
                ['Parameters', r.params],
                ['Training Date', r.training_date],
                ['Testing Date', r.testing_date],
                ['mAP@50', f"{r.map50:.4f}"],
                ['mAP@50-95', f"{r.map50_95:.4f}"],
                ['Precision', f"{r.precision:.4f}"],
                ['Recall', f"{r.recall:.4f}"],
                ['F1 Score', f"{r.f1_score:.4f}"],
                ['Inference FPS', f"{r.fps:.2f}"],
                ['GPU Memory', f"{r.gpu_memory_gb:.2f} GB"],
                ['GPU Temperature', f"{r.gpu_temperature_c:.1f}°C" if r.gpu_temperature_c > 0 else "N/A"],
                ['CPU Usage', f"{r.cpu_percent:.1f}%"],
                ['Training Time', f"{r.training_time_minutes:.1f} min"],
            ]

            detail_table = Table(detail_data, colWidths=[2*inch, 3*inch])
            detail_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ]))
            elements.append(detail_table)
            elements.append(Spacer(1, 20))

        # Build PDF
        doc.build(elements)
        print(f"\n  ✓ Report generated: {report_path}")

        return str(report_path)

    def generate_charts(self):
        """Generate comparison charts"""
        if not self.results:
            return

        charts_dir = self.output_dir / 'charts'
        charts_dir.mkdir(exist_ok=True)

        # Prepare data
        df = pd.DataFrame([asdict(r) for r in self.results])

        # Chart 1: mAP Comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # mAP by model and epochs
        pivot_map50 = df.pivot(index='model_name', columns='epochs', values='map50')
        pivot_map50.plot(kind='bar', ax=axes[0], colormap='viridis')
        axes[0].set_title('mAP@50 by Model and Epochs')
        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('mAP@50')
        axes[0].legend(title='Epochs')
        axes[0].tick_params(axis='x', rotation=45)

        # F1 Score comparison
        pivot_f1 = df.pivot(index='model_name', columns='epochs', values='f1_score')
        pivot_f1.plot(kind='bar', ax=axes[1], colormap='plasma')
        axes[1].set_title('F1 Score by Model and Epochs')
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('F1 Score')
        axes[1].legend(title='Epochs')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(charts_dir / 'model_comparison.png', dpi=150)
        plt.close()

        # Chart 2: FPS vs Accuracy trade-off
        fig, ax = plt.subplots(figsize=(10, 6))

        for epochs in df['epochs'].unique():
            subset = df[df['epochs'] == epochs]
            ax.scatter(subset['fps'], subset['map50'], label=f'{epochs} epochs', s=100, alpha=0.7)
            for _, row in subset.iterrows():
                ax.annotate(row['model_name'], (row['fps'], row['map50']), fontsize=8)

        ax.set_xlabel('Inference FPS')
        ax.set_ylabel('mAP@50')
        ax.set_title('Speed vs Accuracy Trade-off')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(charts_dir / 'speed_accuracy_tradeoff.png', dpi=150)
        plt.close()

        print(f"  ✓ Charts saved to {charts_dir}")

# ============================================================================
# CELL 8: Main Execution
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("  PHASE 1: YOLOv8 BASELINE - PPE DETECTION RESEARCH")
    print("="*70)
    print(f"  Dataset:    {CONFIG['dataset']}")
    print(f"  Image Size: {CONFIG['imgsz']}")
    print(f"  Device:     {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
    print("="*70)

    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download models
    download_models()

    # Initialize profiler
    profiler = HardwareProfiler()

    # Training results
    all_results: List[TrainingResult] = []

    # Determine what to train
    if CONFIG['quick_test']:
        models_to_train = ['yolov8n']
        epochs_to_train = [15]
        print("\n  ⚡ QUICK TEST MODE: Training yolov8n with 15 epochs only\n")
    else:
        models_to_train = list(CONFIG['models'].keys())
        epochs_to_train = CONFIG['epochs_list']

    # Train all configurations
    total_runs = len(models_to_train) * len(epochs_to_train)
    current_run = 0

    for model_name in models_to_train:
        for epochs in epochs_to_train:
            current_run += 1
            print(f"\n  [{current_run}/{total_runs}] Training {model_name} for {epochs} epochs...")

            result = train_model(model_name, epochs, CONFIG, profiler)
            if result:
                all_results.append(result)

    # Save results to JSON
    results_json = output_dir / 'training_results.json'
    with open(results_json, 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\n  ✓ Results saved to {results_json}")

    # Generate report
    if all_results:
        report_gen = PhaseReportGenerator(
            phase=CONFIG['phase'],
            phase_name=CONFIG['phase_name'],
            results=all_results,
            output_dir=str(output_dir)
        )
        report_gen.generate()
        report_gen.generate_charts()

    print("\n" + "="*70)
    print("  PHASE 1 COMPLETE!")
    print("="*70)
    print(f"  Total runs:    {len(all_results)}")
    print(f"  Output dir:    {output_dir}")
    print(f"  Report:        {output_dir}/Phase1_Report.pdf")
    print("="*70 + "\n")

    return all_results

# ============================================================================
# CELL 9: Run
# ============================================================================

if __name__ == '__main__':
    # For quick testing, uncomment the next line:
    # CONFIG['quick_test'] = True

    results = main()
