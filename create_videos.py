#!/usr/bin/env python3
"""
PPE Detection Video Creator
Creates 3 separate videos (LEFT, RIGHT, DOWN) from annotated images
with timestamp overlays and title cards showing EST time
"""

import os
import re
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from ultralytics import YOLO
from tqdm import tqdm

# Configuration
CONFIG = {
    'model_path': 'ppe_yolo8n.pt',
    'confidence': 0.25,
    'video_fps': 15,  # Playback FPS (15 FPS = each 5-sec interval plays in 0.067 sec)
    'time_filter_enabled': True,
    'start_hour_utc': 5,   # 12AM EST
    'end_hour_utc': 21,    # 4PM EST
}

# Paths
BASE_DIR = Path('/Users/hament/Desktop/george aws')
S3_IMAGES_DIR = BASE_DIR / 's3_images'
OUTPUT_DIR = BASE_DIR / 'videos'
ANNOTATED_DIR = BASE_DIR / 'annotated_frames'
CAMERAS = ['left', 'right', 'down']

def parse_timestamp(filename):
    """Extract timestamp from filename"""
    match = re.search(r'image_(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z?', filename)
    if match:
        return datetime.strptime(match.group(1), "%Y-%m-%dT%H:%M:%S")
    return None

def utc_to_est(dt):
    """Convert UTC to EST (UTC-5)"""
    return dt - timedelta(hours=5)

def is_in_time_range(dt, start_hour=5, end_hour=21):
    """Check if UTC hour is in range"""
    if dt is None:
        return True
    return start_hour <= dt.hour < end_hour

def get_sorted_images(folder):
    """Get images sorted by timestamp"""
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG']:
        images.extend(folder.glob(ext))

    # Sort by timestamp in filename
    def sort_key(p):
        ts = parse_timestamp(p.name)
        return ts if ts else datetime.min

    return sorted(images, key=sort_key)

def create_title_card(width, height, camera_name, start_time_est, total_frames):
    """Create a title card image"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)  # Dark gray background

    # Title
    title = f"PPE DETECTION - {camera_name.upper()} CAMERA"
    cv2.putText(img, title, (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)

    # Divider line
    cv2.line(img, (50, 160), (width - 50, 160), (100, 100, 100), 2)

    # Start time
    time_str = start_time_est.strftime("%B %d, %Y")
    cv2.putText(img, "Date:", (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
    cv2.putText(img, time_str, (200, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    time_str2 = start_time_est.strftime("%I:%M:%S %p EST")
    cv2.putText(img, "Start Time:", (50, 290), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
    cv2.putText(img, time_str2, (280, 290), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Frame count
    cv2.putText(img, "Total Frames:", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
    cv2.putText(img, f"{total_frames:,}", (320, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # FPS info
    duration_sec = total_frames / CONFIG['video_fps']
    cv2.putText(img, "Video Duration:", (50, 410), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
    cv2.putText(img, f"{duration_sec:.1f} seconds @ {CONFIG['video_fps']} FPS", (340, 410),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Model info
    cv2.putText(img, "Model: SH17 PPE YOLOv8n", (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 100), 2)
    cv2.putText(img, "Time Filter: 12AM - 4PM EST", (50, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 100), 2)

    return img

def add_timestamp_overlay(img, timestamp_utc, camera_name):
    """Add timestamp overlay to image"""
    if timestamp_utc is None:
        return img

    est_time = utc_to_est(timestamp_utc)

    # Create overlay background
    overlay = img.copy()
    h, w = img.shape[:2]

    # Top bar
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

    # Camera name
    cv2.putText(img, f"{camera_name.upper()} CAMERA", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Timestamp
    time_str = est_time.strftime("%b %d, %Y  %I:%M:%S %p EST")
    cv2.putText(img, time_str, (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    return img

def process_camera(model, camera_name, source_dir, output_video_path):
    """Process all images for one camera and create video"""
    print(f"\n{'='*70}")
    print(f"  PROCESSING {camera_name.upper()} CAMERA")
    print(f"{'='*70}")

    # Get and filter images
    all_images = get_sorted_images(source_dir)
    print(f"  Total images found: {len(all_images):,}")

    if CONFIG['time_filter_enabled']:
        images = [img for img in all_images
                  if is_in_time_range(parse_timestamp(img.name),
                                      CONFIG['start_hour_utc'],
                                      CONFIG['end_hour_utc'])]
        print(f"  After time filter (12AM-4PM EST): {len(images):,}")
    else:
        images = all_images

    if not images:
        print(f"  No images to process!")
        return

    # Get first frame to determine video dimensions
    first_img = cv2.imread(str(images[0]))
    if first_img is None:
        print(f"  Error reading first image!")
        return

    height, width = first_img.shape[:2]
    print(f"  Frame size: {width}x{height}")
    print(f"  Output: {output_video_path}")

    # Get start time for title card
    first_ts = parse_timestamp(images[0].name)
    first_ts_est = utc_to_est(first_ts) if first_ts else datetime.now()

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(output_video_path), fourcc, CONFIG['video_fps'], (width, height))

    # Add title card (show for 3 seconds = 3 * fps frames)
    title_card = create_title_card(width, height, camera_name, first_ts_est, len(images))
    for _ in range(CONFIG['video_fps'] * 3):
        video.write(title_card)

    print(f"\n  Running PPE detection and creating video...")
    print(f"  (This will show annotated frames with timestamp overlays)\n")

    # Process each image
    detections_count = 0
    for img_path in tqdm(images, desc=f"  {camera_name.upper()}", unit="frame"):
        # Run YOLO detection
        results = model.predict(
            source=str(img_path),
            conf=CONFIG['confidence'],
            verbose=False
        )[0]

        # Get annotated image
        annotated = results.plot()

        # Convert BGR to RGB if needed (YOLO returns RGB, OpenCV needs BGR)
        if annotated.shape[2] == 3:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        # Add timestamp overlay
        timestamp = parse_timestamp(img_path.name)
        frame = add_timestamp_overlay(annotated, timestamp, camera_name)

        # Write frame
        video.write(frame)

        # Count detections
        detections_count += len(results.boxes)

    video.release()

    print(f"\n  Video saved: {output_video_path}")
    print(f"  Total detections: {detections_count:,}")
    print(f"  Video duration: {len(images) / CONFIG['video_fps']:.1f} seconds")

def get_date_folders():
    """Get all date folders in s3_images directory"""
    date_folders = []
    if S3_IMAGES_DIR.exists():
        for item in S3_IMAGES_DIR.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if it contains camera subfolders
                has_cameras = any((item / cam).exists() for cam in CAMERAS)
                if has_cameras:
                    date_folders.append(item.name)
    return sorted(date_folders)


def show_available_timestamps():
    """Show available image timestamps for all date folders and cameras"""
    print("\n" + "="*70)
    print("              AVAILABLE S3 IMAGES - TIME RANGES")
    print("="*70)

    date_folders = get_date_folders()
    all_timestamps = []
    folder_info = {}  # Store info for each date/camera combination

    if not date_folders:
        print("\n  No date folders found in s3_images/")
        return all_timestamps, folder_info

    for date_folder in date_folders:
        print(f"\n  {'='*60}")
        print(f"  DATE FOLDER: {date_folder}")
        print(f"  {'='*60}")

        for camera in CAMERAS:
            source_dir = S3_IMAGES_DIR / date_folder / camera
            key = f"{date_folder}/{camera}"

            if not source_dir.exists():
                print(f"\n    {camera.upper()} CAMERA: Folder not found")
                continue

            images = get_sorted_images(source_dir)

            if not images:
                print(f"\n    {camera.upper()} CAMERA: No images found")
                continue

            timestamps = []
            for img in images:
                ts = parse_timestamp(img.name)
                if ts:
                    timestamps.append(ts)
                    all_timestamps.append((date_folder, camera, ts))

            if timestamps:
                first_ts = min(timestamps)
                last_ts = max(timestamps)
                first_est = utc_to_est(first_ts)
                last_est = utc_to_est(last_ts)

                folder_info[key] = {
                    'source_dir': source_dir,
                    'image_count': len(images),
                    'first_ts': first_ts,
                    'last_ts': last_ts,
                }

                print(f"\n    {camera.upper()} CAMERA:")
                print(f"      Total images: {len(images):,}")
                print(f"      First image:  {first_est.strftime('%Y-%m-%d %I:%M:%S %p EST')}")
                print(f"      Last image:   {last_est.strftime('%Y-%m-%d %I:%M:%S %p EST')}")

                # Show time distribution
                hours = {}
                for ts in timestamps:
                    est = utc_to_est(ts)
                    hour_key = est.strftime('%I %p')
                    hours[hour_key] = hours.get(hour_key, 0) + 1

                print(f"      Time distribution (EST):")
                for hour, count in sorted(hours.items()):
                    print(f"        {hour}: {count} images")

    # Overall summary
    if all_timestamps:
        all_ts_only = [ts for _, _, ts in all_timestamps]
        overall_first = min(all_ts_only)
        overall_last = max(all_ts_only)

        print(f"\n  " + "="*60)
        print(f"  SUMMARY:")
        print(f"    Date folders: {', '.join(date_folders)}")
        print(f"    Total videos to create: {len(folder_info)}")
        print(f"    Total images: {len(all_timestamps):,}")
        print(f"    Time range: {utc_to_est(overall_first).strftime('%Y-%m-%d %I:%M %p')} - {utc_to_est(overall_last).strftime('%Y-%m-%d %I:%M %p')} EST")

    return all_timestamps, folder_info


def get_user_time_range():
    """Prompt user to select a time range"""
    print("\n" + "="*70)
    print("              SELECT TIME RANGE (EST)")
    print("="*70)
    print("\n  Enter times in 24-hour format (HH:MM) or press Enter for defaults")
    print("  Current filter: 12:00 AM - 4:00 PM EST\n")

    # Get start time
    start_input = input("  Start time (EST, default 00:00): ").strip()
    if start_input:
        try:
            start_parts = start_input.split(':')
            start_hour_est = int(start_parts[0])
            start_min = int(start_parts[1]) if len(start_parts) > 1 else 0
            # Convert EST to UTC (add 5 hours)
            start_hour_utc = (start_hour_est + 5) % 24
        except:
            print("  Invalid format, using default (00:00 EST = 05:00 UTC)")
            start_hour_utc = 5
    else:
        start_hour_utc = 5  # 12 AM EST

    # Get end time
    end_input = input("  End time (EST, default 16:00): ").strip()
    if end_input:
        try:
            end_parts = end_input.split(':')
            end_hour_est = int(end_parts[0])
            end_min = int(end_parts[1]) if len(end_parts) > 1 else 0
            # Convert EST to UTC (add 5 hours)
            end_hour_utc = (end_hour_est + 5) % 24
        except:
            print("  Invalid format, using default (16:00 EST = 21:00 UTC)")
            end_hour_utc = 21
    else:
        end_hour_utc = 21  # 4 PM EST

    # Convert back to EST for display
    start_est = (start_hour_utc - 5) % 24
    end_est = (end_hour_utc - 5) % 24

    print(f"\n  Selected range: {start_est:02d}:00 EST - {end_est:02d}:00 EST")
    print(f"  (UTC: {start_hour_utc:02d}:00 - {end_hour_utc:02d}:00)")

    return start_hour_utc, end_hour_utc


def main():
    print("="*70)
    print("         PPE DETECTION VIDEO CREATOR")
    print("="*70)

    # Ask user if they want to create videos from S3 images
    print("\n  Do you want to create videos from S3 images?")
    print("  This will process images from date folders containing left, right, and down cameras.")

    response = input("\n  Create videos from S3 images? (yes/no): ").strip().lower()

    if response not in ['yes', 'y']:
        print("\n  Video creation cancelled.")
        return

    # Show available timestamps
    timestamps, folder_info = show_available_timestamps()

    if not timestamps or not folder_info:
        print("\n  No images found in S3 folders. Please download images first.")
        return

    # Ask if user wants to customize time range
    print("\n  " + "-"*60)
    customize = input("\n  Do you want to customize the time range? (yes/no, default: no): ").strip().lower()

    if customize in ['yes', 'y']:
        start_utc, end_utc = get_user_time_range()
        CONFIG['start_hour_utc'] = start_utc
        CONFIG['end_hour_utc'] = end_utc

    # Confirm settings
    print("\n" + "="*70)
    print("              CONFIGURATION")
    print("="*70)
    print(f"\n  Model: {CONFIG['model_path']}")
    print(f"  Video FPS: {CONFIG['video_fps']}")
    start_est = (CONFIG['start_hour_utc'] - 5) % 24
    end_est = (CONFIG['end_hour_utc'] - 5) % 24
    print(f"  Time Filter: {start_est:02d}:00 - {end_est:02d}:00 EST")
    print(f"  Confidence: {CONFIG['confidence']}")
    print(f"\n  Videos to create: {len(folder_info)}")
    for key in sorted(folder_info.keys()):
        print(f"    - {key} ({folder_info[key]['image_count']:,} images)")

    proceed = input("\n  Proceed with video creation? (yes/no): ").strip().lower()
    if proceed not in ['yes', 'y']:
        print("\n  Video creation cancelled.")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load model
    print(f"\n  Loading model...")
    model = YOLO(CONFIG['model_path'])
    print(f"  Model loaded: {len(model.names)} classes")

    # Process each date folder and camera combination
    created_videos = []
    for key, info in sorted(folder_info.items()):
        date_folder, camera = key.split('/')
        source_dir = info['source_dir']
        # Create output filename: ppe_detection_10-31_left_camera.mp4
        output_filename = f'ppe_detection_{date_folder}_{camera}_camera.mp4'
        output_path = OUTPUT_DIR / output_filename

        display_name = f"{date_folder} {camera}"
        process_camera(model, display_name, source_dir, output_path)
        created_videos.append(output_filename)

    print(f"\n{'='*70}")
    print("                    ALL VIDEOS COMPLETE!")
    print(f"{'='*70}")
    print(f"\n  Output files ({len(created_videos)} videos):")
    for video in created_videos:
        print(f"    videos/{video}")
    print()

if __name__ == "__main__":
    main()
