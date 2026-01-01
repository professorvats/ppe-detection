#!/usr/bin/env python3
"""
=============================================================================
PHASE 4: FINE-TUNING & HYPERPARAMETER OPTIMIZATION - PPE DETECTION RESEARCH
=============================================================================
Google Colab Script - Complete Pipeline

This script:
1. Downloads images from AWS S3 (optional)
2. Takes top 3 models from previous phases
3. Performs hyperparameter grid search
4. Optimizes learning rate, augmentation, etc.
5. Generates comprehensive comparison report

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
import itertools
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import subprocess

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

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.enums import TA_CENTER

# ============================================================================
# CELL 2: Configuration
# ============================================================================

CONFIG = {
    'phase': 4,
    'phase_name': 'Fine-tuning & Hyperparameter Optimization',
    'dataset': 'construction-ppe.yaml',
    'imgsz': 640,
    'workers': 4,
    'output_dir': 'ppe_research/phase4_finetuning',

    # Top models to fine-tune (select best from phases 1-3)
    'models_to_tune': {
        'yolo11l': {'batch': 4, 'params': '25.3M', 'size': 'Large'},
        'yolo11m': {'batch': 8, 'params': '20.1M', 'size': 'Medium'},
        'yolo11s': {'batch': 12, 'params': '9.4M', 'size': 'Small'},
    },

    # Base epochs for fine-tuning
    'base_epochs': 100,

    # Hyperparameter search space
    'hyperparams': {
        'lr0': [0.001, 0.01, 0.02],          # Initial learning rate
        'lrf': [0.01, 0.1],                   # Final learning rate factor
        'momentum': [0.9, 0.937, 0.95],       # SGD momentum
        'weight_decay': [0.0001, 0.0005],     # Weight decay
        'warmup_epochs': [1.0, 3.0],          # Warmup epochs
        'mosaic': [0.5, 1.0],                 # Mosaic augmentation
        'mixup': [0.0, 0.1],                  # Mixup augmentation
    },

    # For quick testing - reduced search
    'quick_hyperparams': {
        'lr0': [0.01],
        'lrf': [0.01],
        'momentum': [0.937],
        'weight_decay': [0.0005],
        'warmup_epochs': [3.0],
        'mosaic': [1.0],
        'mixup': [0.0],
    },

    'quick_test': False,
}

# ============================================================================
# CELL 3: Metrics Container
# ============================================================================

@dataclass
class FinetuneMetrics:
    """Metrics for a fine-tuning run"""
    model_name: str = ""
    model_version: str = ""
    model_size: str = ""
    params: str = ""
    epochs: int = 0

    # Hyperparameters used
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    mosaic: float = 1.0
    mixup: float = 0.0

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
    gpu_temperature_c: float = 0.0
    cpu_usage_percent: float = 0.0

    training_time_minutes: float = 0.0

    weights_path: str = ""
    results_dir: str = ""

    # Improvement over baseline
    map50_improvement: float = 0.0


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

        return metrics

    def benchmark(self, model, runs=50) -> Tuple[float, float]:
        dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(10):
            model.predict(dummy, verbose=False)

        latencies = []
        for _ in range(runs):
            start = time.perf_counter()
            model.predict(dummy, verbose=False)
            latencies.append((time.perf_counter() - start) * 1000)

        return 1000 / np.mean(latencies), np.mean(latencies)


# ============================================================================
# CELL 5: Hyperparameter Search
# ============================================================================

def generate_hyperparam_configs(quick: bool = False) -> List[Dict]:
    """Generate all hyperparameter combinations"""
    hp = CONFIG['quick_hyperparams'] if quick else CONFIG['hyperparams']

    keys = list(hp.keys())
    values = list(hp.values())

    configs = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        configs.append(config)

    return configs


def train_with_hyperparams(
    model_name: str,
    hyperparams: Dict,
    profiler: HardwareProfiler,
    run_index: int
) -> Optional[FinetuneMetrics]:
    """Train model with specific hyperparameters"""

    model_config = CONFIG['models_to_tune'][model_name]
    base_model = f"{model_name}.pt"

    # Create unique run name
    hp_str = f"lr{hyperparams['lr0']}_m{hyperparams['momentum']}_wd{hyperparams['weight_decay']}"
    run_name = f"{model_name}_tune{run_index}_{hp_str}_{datetime.now().strftime('%H%M')}"

    size_letter = model_name[-1]
    version = f"v11.0.{size_letter}.tune{run_index}"

    print(f"\n  Run {run_index}: {model_name}")
    print(f"    lr0={hyperparams['lr0']}, momentum={hyperparams['momentum']}")
    print(f"    weight_decay={hyperparams['weight_decay']}, mosaic={hyperparams['mosaic']}")

    training_date = datetime.now().strftime("%b %d, %Y")
    start_time = time.time()

    try:
        model = YOLO(base_model)

        results = model.train(
            data=CONFIG['dataset'],
            epochs=CONFIG['base_epochs'],
            imgsz=CONFIG['imgsz'],
            batch=model_config['batch'],
            patience=15,
            workers=CONFIG['workers'],
            project=CONFIG['output_dir'],
            name=run_name,
            exist_ok=True,
            plots=True,
            save=True,
            verbose=False,
            amp=True,
            # Hyperparameters
            lr0=hyperparams['lr0'],
            lrf=hyperparams['lrf'],
            momentum=hyperparams['momentum'],
            weight_decay=hyperparams['weight_decay'],
            warmup_epochs=hyperparams['warmup_epochs'],
            mosaic=hyperparams['mosaic'],
            mixup=hyperparams['mixup'],
        )

        training_time = (time.time() - start_time) / 60
        testing_date = datetime.now().strftime("%b %d, %Y")

        results_dir = Path(CONFIG['output_dir']) / run_name
        best_weights = results_dir / 'weights' / 'best.pt'

        trained_model = YOLO(str(best_weights))
        val_results = trained_model.val(verbose=False)

        hw = profiler.get_metrics()
        fps, latency = profiler.benchmark(trained_model)

        precision = float(val_results.box.mp)
        recall = float(val_results.box.mr)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics = FinetuneMetrics(
            model_name=model_name,
            model_version=version,
            model_size=model_config['size'],
            params=model_config['params'],
            epochs=CONFIG['base_epochs'],
            lr0=hyperparams['lr0'],
            lrf=hyperparams['lrf'],
            momentum=hyperparams['momentum'],
            weight_decay=hyperparams['weight_decay'],
            warmup_epochs=hyperparams['warmup_epochs'],
            mosaic=hyperparams['mosaic'],
            mixup=hyperparams['mixup'],
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
            gpu_temperature_c=hw['gpu_temperature_c'],
            cpu_usage_percent=hw['cpu_usage_percent'],
            training_time_minutes=training_time,
            weights_path=str(best_weights),
            results_dir=str(results_dir),
        )

        print(f"    ✓ mAP50={metrics.map50:.4f}, F1={metrics.f1_score:.4f}")

        del model, trained_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metrics

    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return None


# ============================================================================
# CELL 6: Report Generator
# ============================================================================

class Phase4ReportGenerator:
    def __init__(self, results: List[FinetuneMetrics], output_dir: str):
        self.results = results
        self.output_dir = Path(output_dir)

    def generate(self) -> str:
        report_path = self.output_dir / "Phase4_Report.pdf"

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
        elements.append(Paragraph("Phase 4: Fine-tuning & Hyperparameter Optimization", title_style))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}", styles['Normal']))
        elements.append(Spacer(1, 15))

        # Best Results Summary
        if self.results:
            best = max(self.results, key=lambda x: x.map50)
            elements.append(Paragraph("Best Configuration Found", heading_style))

            best_data = [
                ['Metric', 'Value'],
                ['Model', best.model_name],
                ['mAP@50', f"{best.map50:.4f}"],
                ['mAP@50-95', f"{best.map50_95:.4f}"],
                ['F1 Score', f"{best.f1_score:.4f}"],
                ['Learning Rate', str(best.lr0)],
                ['Momentum', str(best.momentum)],
                ['Weight Decay', str(best.weight_decay)],
                ['Mosaic', str(best.mosaic)],
                ['Mixup', str(best.mixup)],
                ['FPS', f"{best.fps:.2f}"],
            ]

            best_table = Table(best_data, colWidths=[2*inch, 3*inch])
            best_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
            ]))
            elements.append(best_table)
            elements.append(Spacer(1, 20))

        # All Results Table
        elements.append(Paragraph("All Hyperparameter Configurations", heading_style))

        # Sort by mAP50
        sorted_results = sorted(self.results, key=lambda x: x.map50, reverse=True)

        header = ['Rank', 'Model', 'lr0', 'momentum', 'w_decay', 'mosaic', 'mAP50', 'mAP50-95', 'F1', 'FPS']
        table_data = [header]

        for i, r in enumerate(sorted_results[:20], 1):  # Top 20
            table_data.append([
                str(i),
                r.model_name,
                str(r.lr0),
                str(r.momentum),
                str(r.weight_decay),
                str(r.mosaic),
                f"{r.map50:.4f}",
                f"{r.map50_95:.4f}",
                f"{r.f1_score:.4f}",
                f"{r.fps:.1f}",
            ])

        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#d5f5e3')),  # Highlight best
        ]))
        elements.append(table)

        doc.build(elements)
        print(f"\n  ✓ Report: {report_path}")
        return str(report_path)

    def save_json(self):
        json_path = self.output_dir / "finetuning_results.json"
        with open(json_path, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        print(f"  ✓ JSON: {json_path}")

    def generate_charts(self):
        if not self.results:
            return

        df = pd.DataFrame([asdict(r) for r in self.results])

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Learning rate impact
        lr_grouped = df.groupby('lr0')['map50'].mean()
        axes[0, 0].bar(lr_grouped.index.astype(str), lr_grouped.values, color='steelblue')
        axes[0, 0].set_title('mAP@50 by Learning Rate')
        axes[0, 0].set_xlabel('Learning Rate')
        axes[0, 0].set_ylabel('mAP@50')

        # Momentum impact
        mom_grouped = df.groupby('momentum')['map50'].mean()
        axes[0, 1].bar(mom_grouped.index.astype(str), mom_grouped.values, color='coral')
        axes[0, 1].set_title('mAP@50 by Momentum')
        axes[0, 1].set_xlabel('Momentum')

        # Mosaic impact
        mosaic_grouped = df.groupby('mosaic')['map50'].mean()
        axes[1, 0].bar(mosaic_grouped.index.astype(str), mosaic_grouped.values, color='seagreen')
        axes[1, 0].set_title('mAP@50 by Mosaic Augmentation')
        axes[1, 0].set_xlabel('Mosaic')

        # Top 10 configs
        top10 = df.nlargest(10, 'map50')
        axes[1, 1].barh(range(len(top10)), top10['map50'].values, color='purple')
        axes[1, 1].set_yticks(range(len(top10)))
        axes[1, 1].set_yticklabels([f"{r['model_name']}\nlr={r['lr0']}" for _, r in top10.iterrows()], fontsize=7)
        axes[1, 1].set_title('Top 10 Configurations')
        axes[1, 1].set_xlabel('mAP@50')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'phase4_analysis.png', dpi=150)
        plt.close()
        print(f"  ✓ Charts saved")


# ============================================================================
# CELL 7: Main
# ============================================================================

def main():
    print("\n" + "="*70)
    print("  PHASE 4: FINE-TUNING & HYPERPARAMETER OPTIMIZATION")
    print("="*70)
    print(f"  Device: {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Models: {list(CONFIG['models_to_tune'].keys())}")
    print("="*70)

    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    profiler = HardwareProfiler()

    # Generate hyperparameter configurations
    hp_configs = generate_hyperparam_configs(quick=CONFIG['quick_test'])
    print(f"\n  Hyperparameter configurations: {len(hp_configs)}")

    models = list(CONFIG['models_to_tune'].keys())
    total_runs = len(models) * len(hp_configs)
    print(f"  Total runs: {total_runs}")

    if CONFIG['quick_test']:
        print("\n  ⚡ QUICK TEST MODE\n")

    all_results = []
    run_index = 0

    for model_name in models:
        print(f"\n  === Tuning {model_name} ===")
        for hp_config in hp_configs:
            run_index += 1
            print(f"\n  [{run_index}/{total_runs}]")
            result = train_with_hyperparams(model_name, hp_config, profiler, run_index)
            if result:
                all_results.append(result)

    if all_results:
        report_gen = Phase4ReportGenerator(all_results, str(output_dir))
        report_gen.save_json()
        report_gen.generate_charts()
        report_gen.generate()

        # Print best configuration
        best = max(all_results, key=lambda x: x.map50)
        print("\n" + "="*70)
        print("  BEST CONFIGURATION FOUND:")
        print(f"    Model: {best.model_name}")
        print(f"    mAP@50: {best.map50:.4f}")
        print(f"    lr0: {best.lr0}, momentum: {best.momentum}")
        print(f"    weight_decay: {best.weight_decay}, mosaic: {best.mosaic}")
        print("="*70)

    print("\n  PHASE 4 COMPLETE!")
    print(f"  Results: {len(all_results)} runs")
    print(f"  Report: {output_dir}/Phase4_Report.pdf\n")

    return all_results


if __name__ == '__main__':
    # CONFIG['quick_test'] = True
    results = main()
