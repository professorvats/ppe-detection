#!/usr/bin/env python3
"""
PPE Detection Report Generator
Run: python ppe_report.py

Creates numbered reports: report_1_from_22-Dec-0108PM, report_2_from_...
Filters for images with 3+ persons detected (more than 2)
Shows samples from all 3 cameras (left, right, down)

Enhanced features:
- Fish-eye lens correction
- YOLOv11 model support
- Class-specific confidence thresholds
- Adaptive PPE association margins
- Minimum bounding box size filtering
"""

import os
import re
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from tqdm import tqdm

# Import lens correction (optional, gracefully degrade if not available)
try:
    from lens_correction import LensCorrector
    LENS_CORRECTION_AVAILABLE = True
except ImportError:
    LENS_CORRECTION_AVAILABLE = False
    print("  Note: lens_correction module not found. Running without lens correction.")

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    'time_filter_enabled': True,
    'start_hour_utc': 5,      # 12AM EST (fallback)
    'end_hour_utc': 21,       # 4PM EST (fallback)
    'base_confidence': 0.15,  # Base confidence (class-specific thresholds applied after)
    'model_path': 'ppe_yolo11l_best.pt',  # YOLOv11 large model (fallback to yolo8n if not found)
    'min_persons': 1,         # Minimum persons to include in report samples
    'max_samples_per_camera': 50,  # Max sample images per camera per date

    # Lens correction
    'lens_correction_enabled': False,  # Disabled - was distorting images
    'lens_correction_config': 'camera_calibration.json',

    # Image preprocessing
    'apply_clahe': False,           # Disabled - use original image quality
    'normalize_brightness': False,  # Disabled - use original image quality

    # Date-specific time windows (EST converted to UTC by adding 5 hours)
    'date_time_windows': {
        '10-31': {'start': '17:10:00', 'end': '18:10:50'},  # 12:10 PM - 1:10:50 PM EST
        '11-7': {'start': '18:07:55', 'end': '18:35:42'},   # 1:07:55 PM - 1:35:42 PM EST
    },
}

# Class-specific confidence thresholds (applied after detection)
CLASS_CONFIDENCE_THRESHOLDS = {
    'person': 0.20,        # Lower for better recall (catch more people)
    'helmet': 0.35,
    'safety-vest': 0.30,
    'gloves': 0.40,
    'glasses': 0.45,
    'face-mask': 0.40,
    'face-guard': 0.40,
    'ear-mufs': 0.45,
    'safety-suit': 0.35,
    'Hardhat': 0.35,       # Dataset uses these names
    'Safety Vest': 0.30,
    'NO-Hardhat': 0.50,    # Higher threshold for violations
    'NO-Safety Vest': 0.50,
    'Person': 0.20,
    'default': 0.30,
}

# Minimum detection sizes (filter out noise/tiny detections)
MIN_DETECTION_SIZE = {
    'person': {'min_width': 30, 'min_height': 60, 'min_area': 2000},
    'Person': {'min_width': 30, 'min_height': 60, 'min_area': 2000},
    'helmet': {'min_width': 15, 'min_height': 15, 'min_area': 300},
    'Hardhat': {'min_width': 15, 'min_height': 15, 'min_area': 300},
    'safety-vest': {'min_width': 20, 'min_height': 25, 'min_area': 600},
    'Safety Vest': {'min_width': 20, 'min_height': 25, 'min_area': 600},
    'default': {'min_width': 10, 'min_height': 10, 'min_area': 150},
}

# Required PPE for compliance check
REQUIRED_PPE = {'helmet', 'safety-vest', 'gloves', 'glasses', 'safety-suit', 'Hardhat', 'Safety Vest'}
DETECTED_PPE = {'helmet', 'safety-vest', 'gloves', 'glasses', 'face-mask', 'face-guard', 'ear-mufs', 'safety-suit',
                'Hardhat', 'Safety Vest', 'Mask'}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_report_folder():
    """Create report folder like: report_1_from_22-Dec-0108PM"""
    base = Path('.')
    existing = list(base.glob('report_*'))

    # Find next number
    nums = []
    for p in existing:
        match = re.match(r'report_(\d+)_', p.name)
        if match:
            nums.append(int(match.group(1)))

    next_num = max(nums) + 1 if nums else 1

    # Create timestamp
    now = datetime.now()
    timestamp = now.strftime('%d-%b-%I%M%p')  # 22-Dec-0108PM

    folder_name = f'report_{next_num}_from_{timestamp}'
    folder = Path(folder_name)
    folder.mkdir(exist_ok=True)

    return folder

def parse_timestamp(filename):
    """Extract timestamp from filename"""
    match = re.search(r'image_(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)', filename)
    return match.group(1) if match else None

def is_in_time_range(ts, start=5, end=21):
    """Check if timestamp is in allowed range"""
    if not ts:
        return True  # Include if no timestamp
    try:
        fmt = "%Y-%m-%dT%H:%M:%S.%fZ" if '.' in ts else "%Y-%m-%dT%H:%M:%SZ"
        return start <= datetime.strptime(ts, fmt).hour < end
    except:
        return True

def get_images(folder):
    """Get all images from folder"""
    p = Path(folder)
    if not p.exists():
        return []
    imgs = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG']:
        imgs.extend(p.glob(ext))
    return sorted([str(x) for x in imgs])

def filter_by_time(paths, start=5, end=21):
    """Filter images by time range"""
    filtered = []
    for p in paths:
        ts = parse_timestamp(Path(p).name)
        if ts is None or is_in_time_range(ts, start, end):
            filtered.append(p)
    return filtered

def filter_by_date_time_window(paths, date_key, time_windows):
    """Filter images by date-specific time windows"""
    if date_key not in time_windows:
        return paths  # No filter for this date

    window = time_windows[date_key]
    start_time = window['start']  # Format: "HH:MM:SS"
    end_time = window['end']

    filtered = []
    for p in paths:
        ts = parse_timestamp(Path(p).name)
        if ts is None:
            filtered.append(p)
            continue
        try:
            fmt = "%Y-%m-%dT%H:%M:%S.%fZ" if '.' in ts else "%Y-%m-%dT%H:%M:%SZ"
            dt = datetime.strptime(ts, fmt)
            img_time = dt.strftime("%H:%M:%S")
            if start_time <= img_time <= end_time:
                filtered.append(p)
        except:
            filtered.append(p)
    return filtered


# ============================================================
# DETECTION IMPROVEMENT FUNCTIONS
# ============================================================

def filter_detections_by_class_threshold(boxes, names, thresholds):
    """
    Filter YOLO detections using class-specific confidence thresholds.

    Args:
        boxes: YOLO result.boxes
        names: result.names mapping
        thresholds: dict mapping class names to confidence thresholds

    Returns:
        List of indices for boxes that pass threshold
    """
    valid_indices = []

    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]
        conf = float(box.conf[0])

        # Get threshold for this class (fallback to default)
        threshold = thresholds.get(cls_name, thresholds.get('default', 0.30))

        if conf >= threshold:
            valid_indices.append(i)

    return valid_indices


def filter_by_minimum_size(detections, size_config):
    """
    Filter out detections below minimum size thresholds.

    Args:
        detections: list of detection dicts with 'box' and 'class' keys
        size_config: dict with min size requirements per class

    Returns:
        Filtered list of detections
    """
    filtered = []

    for det in detections:
        x1, y1, x2, y2 = det['box']
        width = x2 - x1
        height = y2 - y1
        area = width * height

        cls_name = det['class']
        constraints = size_config.get(cls_name, size_config.get('default', {}))

        min_width = constraints.get('min_width', 10)
        min_height = constraints.get('min_height', 10)
        min_area = constraints.get('min_area', 100)

        if width >= min_width and height >= min_height and area >= min_area:
            filtered.append(det)

    return filtered


def calculate_adaptive_margin(person_box, image_shape, base_margin_ratio=0.15):
    """
    Calculate adaptive margin based on person bounding box size.

    Args:
        person_box: tuple (x1, y1, x2, y2) of person bounding box
        image_shape: tuple (height, width) of image
        base_margin_ratio: ratio of person bbox size to use as margin

    Returns:
        tuple (margin_x, margin_y) - horizontal and vertical margins
    """
    x1, y1, x2, y2 = person_box
    person_width = x2 - x1
    person_height = y2 - y1

    # Calculate margin as percentage of person size (minimum 30px)
    margin_x = max(30, int(person_width * base_margin_ratio))
    margin_y = max(30, int(person_height * base_margin_ratio))

    # Cap margin to prevent excessive expansion (10% of image)
    img_h, img_w = image_shape[:2]
    max_margin_x = int(img_w * 0.10)
    max_margin_y = int(img_h * 0.10)

    margin_x = min(margin_x, max_margin_x)
    margin_y = min(margin_y, max_margin_y)

    return margin_x, margin_y


def preprocess_image_for_detection(img, apply_clahe=True, normalize_brightness=True):
    """
    Preprocess image to improve detection accuracy.

    Args:
        img: input BGR image (numpy array)
        apply_clahe: apply Contrast Limited Adaptive Histogram Equalization
        normalize_brightness: normalize image brightness

    Returns:
        Preprocessed image (numpy array)
    """
    if img is None:
        return None

    if not apply_clahe and not normalize_brightness:
        return img

    # Convert to LAB color space for better preprocessing
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    if apply_clahe:
        # Apply CLAHE to L channel for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

    if normalize_brightness:
        # Normalize brightness to target mean (120)
        target_mean = 120
        current_mean = np.mean(l)
        if current_mean < 80 or current_mean > 180:  # Only adjust if needed
            scale = target_mean / max(current_mean, 1)
            l = np.clip(l * scale, 0, 255).astype(np.uint8)

    # Merge and convert back to BGR
    lab = cv2.merge([l, a, b])
    processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return processed


def annotate_image(img_path, result, output_path, image_for_annotation=None):
    """
    Custom annotation with thin borders and PPE status per person.
    Uses class-specific thresholds, adaptive margins, and size filtering.

    Args:
        img_path: Path to original image
        result: YOLO prediction result
        output_path: Where to save annotated image
        image_for_annotation: Optional pre-loaded image (for lens-corrected images)

    Returns: number of persons detected
    """
    # Use provided image or load from path
    if image_for_annotation is not None:
        img = image_for_annotation.copy()
    else:
        img = cv2.imread(str(img_path))

    if img is None:
        return 0

    image_shape = img.shape  # (height, width, channels)

    # Get valid detection indices using class-specific thresholds
    valid_indices = filter_detections_by_class_threshold(
        result.boxes, result.names, CLASS_CONFIDENCE_THRESHOLDS
    )

    # Separate persons from PPE items (only valid detections)
    persons = []
    ppe_items = []

    for i in valid_indices:
        box = result.boxes[i]
        cls_id = int(box.cls[0])
        cls_name = result.names[cls_id]
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, xyxy)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        item = {'class': cls_name, 'conf': conf, 'box': (x1, y1, x2, y2), 'center': (cx, cy)}

        if cls_name.lower() == 'person':
            persons.append(item)
        elif cls_name in DETECTED_PPE:
            ppe_items.append(item)

    # Apply minimum size filtering
    persons = filter_by_minimum_size(persons, MIN_DETECTION_SIZE)
    ppe_items = filter_by_minimum_size(ppe_items, MIN_DETECTION_SIZE)

    # Associate PPE with persons using adaptive margins
    for person in persons:
        px1, py1, px2, py2 = person['box']
        person['ppe'] = []

        # Calculate adaptive margin based on person size
        margin_x, margin_y = calculate_adaptive_margin(person['box'], image_shape)

        for ppe in ppe_items:
            pcx, pcy = ppe['center']
            if (px1 - margin_x <= pcx <= px2 + margin_x) and (py1 - margin_y <= pcy <= py2 + margin_y):
                person['ppe'].append(ppe['class'])

    # Colors
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLACK = (0, 0, 0)

    # Draw annotations
    for person in persons:
        x1, y1, x2, y2 = person['box']
        conf = person['conf']

        # Thin border (1px)
        cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 1)

        # Labels
        label1 = f"person {conf*100:.0f}%"
        has_ppe = set(person['ppe'])
        missing = REQUIRED_PPE - has_ppe

        if missing:
            label2 = f"No: {', '.join(sorted(missing))}"
            color2 = RED
        else:
            label2 = "PPE OK"
            color2 = GREEN

        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.4
        thick = 1

        (tw1, th1), _ = cv2.getTextSize(label1, font, scale, thick)
        (tw2, th2), _ = cv2.getTextSize(label2, font, scale, thick)

        # Position text
        ty1 = y1 + th1 + 4
        ty2 = ty1 + th2 + 4

        if ty2 > y2:
            ty1 = y1 - th2 - th1 - 8
            ty2 = y1 - 4
            if ty1 < 0:
                ty1 = y1 + th1 + 4
                ty2 = ty1 + th2 + 4

        # Draw labels
        cv2.rectangle(img, (x1, ty1 - th1 - 2), (x1 + tw1 + 4, ty1 + 2), BLACK, -1)
        cv2.putText(img, label1, (x1 + 2, ty1), font, scale, GREEN, thick)

        cv2.rectangle(img, (x1, ty2 - th2 - 2), (x1 + tw2 + 4, ty2 + 2), BLACK, -1)
        cv2.putText(img, label2, (x1 + 2, ty2), font, scale, color2, thick)

    # Save with high JPEG quality (95%)
    cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return len(persons)

def process_camera(model, images, camera_name, report_folder, min_persons=2, max_samples=10,
                    lens_corrector=None):
    """
    Process images from one camera with lens correction and improved detection.

    Args:
        model: YOLO model
        images: List of image paths
        camera_name: Camera identifier ('left', 'right', 'down')
        report_folder: Output folder for annotated images
        min_persons: Minimum persons for sample selection
        max_samples: Maximum sample images to collect
        lens_corrector: Optional LensCorrector instance

    Returns:
        detections dataframe, list of sample images with 3+ persons
    """
    output_dir = report_folder / camera_name
    output_dir.mkdir(exist_ok=True)

    detections = []
    multi_person_samples = []

    print(f"\n  Processing {camera_name}: {len(images):,} images")
    if lens_corrector and lens_corrector.is_enabled():
        print(f"    Lens correction: ENABLED")
    if CONFIG.get('apply_clahe') or CONFIG.get('normalize_brightness'):
        print(f"    Image preprocessing: ENABLED")

    for img_path in tqdm(images, desc=f"  {camera_name}", unit="img"):
        img_name = Path(img_path).name
        ts = parse_timestamp(img_name)

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            detections.append({
                'image': img_name, 'camera': camera_name, 'timestamp': ts,
                'class': 'LOAD_ERROR', 'confidence': None, 'persons_in_image': 0
            })
            continue

        # Apply lens correction
        if lens_corrector and lens_corrector.is_enabled():
            img = lens_corrector.undistort(img, camera_name)

        # Apply preprocessing (CLAHE, brightness normalization)
        img_for_detection = preprocess_image_for_detection(
            img,
            apply_clahe=CONFIG.get('apply_clahe', False),
            normalize_brightness=CONFIG.get('normalize_brightness', False)
        )

        # Run detection on preprocessed image (half=True for 2x speed on GPU)
        # Only detect: Person (6), helmet (0), vest (2)
        result = model.predict(
            source=img_for_detection,
            save=False,
            conf=CONFIG.get('base_confidence', 0.15),
            verbose=False,
            half=True,
            device=0,
            classes=[0, 2, 6]
        )[0]

        # Count persons using class-specific thresholds
        valid_indices = filter_detections_by_class_threshold(
            result.boxes, result.names, CLASS_CONFIDENCE_THRESHOLDS
        )
        person_count = sum(
            1 for i in valid_indices
            if result.names[int(result.boxes[i].cls[0])].lower() == 'person'
        )

        # Save annotated image (use original/corrected image, not preprocessed)
        output_path = output_dir / img_name
        annotate_image(img_path, result, output_path, image_for_annotation=img)

        # Track multi-person images for samples
        if person_count >= min_persons and len(multi_person_samples) < max_samples:
            multi_person_samples.append({
                'path': str(output_path),
                'name': img_name,
                'persons': person_count,
                'timestamp': ts
            })

        # Record detections (only those passing class-specific thresholds)
        if len(valid_indices) == 0:
            detections.append({
                'image': img_name, 'camera': camera_name, 'timestamp': ts,
                'class': 'NO_DETECTION', 'confidence': None, 'persons_in_image': 0
            })
        else:
            for i in valid_indices:
                box = result.boxes[i]
                cls_name = result.names[int(box.cls[0])]
                conf = round(float(box.conf[0]), 4)
                detections.append({
                    'image': img_name, 'camera': camera_name, 'timestamp': ts,
                    'class': cls_name, 'confidence': conf, 'persons_in_image': person_count
                })

    return pd.DataFrame(detections), multi_person_samples

def generate_html_report(report_folder, samples_by_camera, stats, class_counts, all_df=None):
    """Generate HTML report with sample images and detailed analysis"""

    # Calculate additional stats from dataframe
    date_stats = {}
    compliance_stats = {'with_helmet': 0, 'without_helmet': 0, 'with_vest': 0, 'without_vest': 0}

    if all_df is not None and len(all_df) > 0:
        # Per-date breakdown
        for date_folder in all_df['date_folder'].unique():
            date_data = all_df[all_df['date_folder'] == date_folder]
            date_detections = date_data[date_data['class'] != 'NO_DETECTION']
            date_stats[date_folder] = {
                'images': len(date_data['image'].unique()),
                'detections': len(date_detections),
                'persons': len(date_data[date_data['class'] == 'Person']),
                'helmets': len(date_data[date_data['class'] == 'helmet']),
                'vests': len(date_data[date_data['class'] == 'vest']),
            }

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>PPE Detection Report - {report_folder.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }}
        h1 {{ color: #4CAF50; }}
        h2 {{ color: #2196F3; border-bottom: 1px solid #333; padding-bottom: 10px; }}
        h3 {{ color: #FF9800; }}
        .stats {{ background: #2a2a2a; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}
        .stats-grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }}
        .stat-box {{ background: #333; padding: 15px; border-radius: 5px; text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
        .stat-value.warning {{ color: #FF9800; }}
        .stat-value.danger {{ color: #f44336; }}
        .stat-label {{ color: #888; font-size: 12px; }}
        .camera-section {{ margin: 30px 0; }}
        .samples {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }}
        .sample {{ background: #2a2a2a; padding: 10px; border-radius: 8px; }}
        .sample img {{ width: 100%; border-radius: 5px; cursor: pointer; transition: transform 0.2s; }}
        .sample img:hover {{ transform: scale(1.02); }}
        .sample-info {{ padding: 10px 0; font-size: 11px; color: #888; }}
        .sample-info strong {{ color: #fff; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #333; }}
        th {{ background: #333; color: #4CAF50; }}
        tr:hover {{ background: #2a2a2a; }}
        .class-table {{ max-width: 500px; }}
        .date-section {{ background: #252525; padding: 15px; border-radius: 8px; margin: 15px 0; }}
        .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 11px; margin: 2px; }}
        .badge-person {{ background: #2196F3; }}
        .badge-helmet {{ background: #4CAF50; }}
        .badge-vest {{ background: #FF9800; }}
    </style>
</head>
<body>
    <h1>PPE Detection Analysis Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>Report: {report_folder.name}</p>
    <p>Camera: <strong>DOWN</strong> | Classes: <strong>Person, Helmet, Vest</strong></p>

    <div class="stats">
        <h2>Overall Summary</h2>
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value">{stats['total_images']:,}</div>
                <div class="stat-label">Total Images Analyzed</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{stats['total_detections']:,}</div>
                <div class="stat-label">Total Detections</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{stats['images_with_persons']:,}</div>
                <div class="stat-label">Images with Persons</div>
            </div>
        </div>

        <h3>Detection Breakdown by Class</h3>
        <div class="stats-grid-4">
            <div class="stat-box">
                <div class="stat-value">{class_counts.get('Person', 0):,}</div>
                <div class="stat-label"><span class="badge badge-person">Person</span></div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{class_counts.get('helmet', 0):,}</div>
                <div class="stat-label"><span class="badge badge-helmet">Helmet</span></div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{class_counts.get('vest', 0):,}</div>
                <div class="stat-label"><span class="badge badge-vest">Vest</span></div>
            </div>
            <div class="stat-box">
                <div class="stat-value {'warning' if class_counts.get('Person', 0) > 0 and class_counts.get('helmet', 0) < class_counts.get('Person', 0) else ''}">{class_counts.get('Person', 0) - class_counts.get('helmet', 0) if class_counts.get('Person', 0) > class_counts.get('helmet', 0) else 0}</div>
                <div class="stat-label">Persons Without Helmet</div>
            </div>
        </div>
    </div>
"""

    # Add per-date breakdown
    if date_stats:
        html += """
    <div class="stats">
        <h2>Per-Date Analysis</h2>
"""
        for date_folder, ds in date_stats.items():
            helmet_ratio = (ds['helmets'] / ds['persons'] * 100) if ds['persons'] > 0 else 0
            vest_ratio = (ds['vests'] / ds['persons'] * 100) if ds['persons'] > 0 else 0
            html += f"""
        <div class="date-section">
            <h3>{date_folder}</h3>
            <div class="stats-grid-4">
                <div class="stat-box">
                    <div class="stat-value">{ds['images']:,}</div>
                    <div class="stat-label">Images</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{ds['persons']:,}</div>
                    <div class="stat-label">Persons Detected</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{ds['helmets']:,}</div>
                    <div class="stat-label">Helmets ({helmet_ratio:.0f}%)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{ds['vests']:,}</div>
                    <div class="stat-label">Vests ({vest_ratio:.0f}%)</div>
                </div>
            </div>
        </div>
"""
        html += """
    </div>
"""

    # Add camera samples (grouped by date folder)
    for camera, samples in samples_by_camera.items():
        if not samples:
            continue  # Skip empty camera sections

        html += f"""
    <div class="camera-section">
        <h2>{camera.upper()} Camera - Images with {CONFIG['min_persons']}+ Persons</h2>
        <div class="samples">
"""
        for sample in samples:
            # Path includes date_folder: {date_folder}/{camera}/{name}
            date_folder = sample.get('date_folder', '')
            rel_path = f"{date_folder}/{camera}/{sample['name']}"
            html += f"""
            <div class="sample">
                <img src="{rel_path}" alt="{sample['name']}">
                <div class="sample-info">
                    <strong>{sample['name']}</strong><br>
                    Date: {date_folder} | Persons: {sample['persons']} | Time: {sample['timestamp'] or 'N/A'}
                </div>
            </div>
"""

        html += """
        </div>
    </div>
"""

    html += """
</body>
</html>
"""

    report_path = report_folder / 'report.html'
    with open(report_path, 'w') as f:
        f.write(html)

    return report_path

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("  PPE DETECTION REPORT GENERATOR (Enhanced)")
    print("=" * 70)

    # Create report folder
    report_folder = get_report_folder()
    print(f"\n  Report folder: {report_folder}/")

    # Initialize lens corrector
    lens_corrector = None
    if LENS_CORRECTION_AVAILABLE and CONFIG.get('lens_correction_enabled', False):
        config_path = CONFIG.get('lens_correction_config', 'camera_calibration.json')
        if Path(config_path).exists():
            lens_corrector = LensCorrector(config_path)
            print(f"\n  Lens correction: ENABLED ({config_path})")
        else:
            print(f"\n  Lens correction: DISABLED (config not found: {config_path})")
    else:
        print(f"\n  Lens correction: DISABLED")

    # Load model (with fallback)
    print("\n  Loading model...")
    model_path = CONFIG['model_path']

    # Try primary model, fallback to alternatives if not found
    if not Path(model_path).exists():
        fallback_models = ['ppe_yolo11m_best.pt', 'ppe_yolov8_best.pt', 'ppe_yolo8n.pt', 'yolo11l.pt']
        for fallback in fallback_models:
            if Path(fallback).exists():
                print(f"  Primary model not found: {model_path}")
                model_path = fallback
                break

    model = YOLO(model_path)
    print(f"  Model loaded: {model_path}")
    print(f"  Classes: {list(model.names.values())}")

    # Get images from ALL date folders
    print("\n  Scanning images...")
    date_folders = ['10-31', '11-7']

    all_dfs = []
    all_left_samples = []
    all_right_samples = []
    all_down_samples = []
    total_stats = {'left': 0, 'right': 0, 'down': 0}

    for date_folder in date_folders:
        base_path = f's3_images/{date_folder}'
        if not Path(base_path).exists():
            print(f"  Skipping {date_folder} (folder not found)")
            continue

        # Only process DOWN camera
        down_all = get_images(f'{base_path}/down')

        print(f"\n  {date_folder}: Down={len(down_all):,} images")

        # Filter by date-specific time windows
        if CONFIG['time_filter_enabled'] and 'date_time_windows' in CONFIG:
            date_key = date_folder
            down = filter_by_date_time_window(down_all, date_key, CONFIG['date_time_windows'])
            window = CONFIG['date_time_windows'].get(date_key, {})
            print(f"    After time filter ({window.get('start', 'N/A')} - {window.get('end', 'N/A')} UTC): Down={len(down):,}")
        else:
            down = down_all

        if len(down) == 0:
            print(f"    No images in time window, skipping...")
            continue

        total_stats['down'] += len(down)

        # Process down camera for this date
        print(f"\n  Processing {date_folder} (down camera only)...")

        # Create date-specific subfolders
        date_report_folder = report_folder / date_folder
        date_report_folder.mkdir(exist_ok=True)

        down_df, down_samples = process_camera(model, down, 'down', date_report_folder,
                                               CONFIG['min_persons'], CONFIG['max_samples_per_camera'],
                                               lens_corrector=lens_corrector)

        # Add date column to track which folder images came from
        down_df['date_folder'] = date_folder

        # Update sample paths to include date folder
        for s in down_samples:
            s['date_folder'] = date_folder

        all_dfs.append(down_df)
        all_down_samples.extend(down_samples)

    # Combine results from all dates
    all_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    left_samples = []  # Not processing left/right
    right_samples = []
    down_samples = all_down_samples
    all_df.to_csv(report_folder / 'all_detections.csv', index=False)

    # Calculate stats
    if len(all_df) > 0:
        actual_detections = all_df[all_df['class'] != 'NO_DETECTION']
        class_counts = actual_detections['class'].value_counts().to_dict()
    else:
        actual_detections = pd.DataFrame()
        class_counts = {}

    stats = {
        'total_images': total_stats['down'],  # Only down camera
        'total_detections': len(actual_detections),
        'images_with_persons': len(all_df[all_df['persons_in_image'] > 0]['image'].unique()) if len(all_df) > 0 else 0,
        'left_images': 0,
        'right_images': 0,
        'down_images': total_stats['down'],
    }

    # Generate HTML report
    samples_by_camera = {
        'left': left_samples,
        'right': right_samples,
        'down': down_samples
    }

    html_path = generate_html_report(report_folder, samples_by_camera, stats, class_counts, all_df)

    # Save JSON summary
    summary = {
        'generated': datetime.now().isoformat(),
        'report_folder': str(report_folder),
        'config': CONFIG,
        'stats': stats,
        'class_counts': class_counts,
        'samples': {
            'left': len(left_samples),
            'right': len(right_samples),
            'down': len(down_samples)
        }
    }
    with open(report_folder / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print(f"  REPORT COMPLETE: {report_folder}/")
    print("=" * 70)
    print(f"\n  Total images processed: {stats['total_images']:,}")
    print(f"  Total detections: {stats['total_detections']:,}")
    print(f"  Images with persons: {stats['images_with_persons']:,}")

    print(f"\n  Samples with {CONFIG['min_persons']}+ persons:")
    print(f"    Left camera:  {len(left_samples)} samples")
    print(f"    Right camera: {len(right_samples)} samples")
    print(f"    Down camera:  {len(down_samples)} samples")

    print(f"\n  Output files:")
    print(f"    {report_folder}/report.html        <- Open this in browser!")
    print(f"    {report_folder}/all_detections.csv")
    print(f"    {report_folder}/summary.json")
    print(f"    {report_folder}/left/   (annotated images)")
    print(f"    {report_folder}/right/  (annotated images)")
    print(f"    {report_folder}/down/   (annotated images)")

    print("\n" + "=" * 70)
    print(f"  Run again to create next report!")
    print("=" * 70)

if __name__ == '__main__':
    main()
