#!/usr/bin/env python3
"""Download images from S3 with progress bar."""

import boto3
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

BUCKET = "sst-steeleye-connect-api-prod-device-uploads"
BASE_PREFIX = "uploads/5036651f-bd4a-4f41-8759-0848908a7db5/7c41d851-4e19-4aa8-9cb7-fd7a3efab614/"

# Dates to download with their folder names
DATES = [
    ("2025-10-31", "10-31"),
    ("2025-11-07", "11-7"),
]

MAX_WORKERS = 50

# Thread-local storage for S3 clients (boto3 clients aren't thread-safe)
thread_local = threading.local()

def get_s3_client():
    if not hasattr(thread_local, 's3'):
        thread_local.s3 = boto3.client('s3')
    return thread_local.s3

def download_single_file(args):
    key, local_file = args
    local_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3 = get_s3_client()
        s3.download_file(BUCKET, key, str(local_file))
        return True, key
    except Exception as e:
        return False, f"{key}: {e}"

def download_s3_folder(s3_date, local_name):
    s3 = boto3.client('s3')
    prefix = f"{BASE_PREFIX}{s3_date}/"
    local_path = Path(f"s3_images/{local_name}")
    local_path.mkdir(parents=True, exist_ok=True)

    # Get list of all objects
    print(f"\nScanning s3://{BUCKET}/{prefix}...")
    download_tasks = []
    paginator = s3.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            relative_path = key[len(prefix):].lstrip('/')
            if not relative_path:
                continue
            local_file = local_path / relative_path
            download_tasks.append((key, local_file))

    if not download_tasks:
        print("No files found to download")
        return

    print(f"Found {len(download_tasks)} files to download (using {MAX_WORKERS} threads)")

    # Download concurrently with progress bar
    errors = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_single_file, task): task for task in download_tasks}
        with tqdm(total=len(futures), desc=f"Downloading {local_name}", unit="file") as pbar:
            for future in as_completed(futures):
                success, result = future.result()
                if not success:
                    errors.append(result)
                pbar.update(1)

    if errors:
        print(f"\n{len(errors)} errors occurred:")
        for err in errors[:5]:
            print(f"  - {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    print(f"Done! Files saved to: s3_images/{local_name}")

if __name__ == "__main__":
    for s3_date, local_name in DATES:
        download_s3_folder(s3_date, local_name)
    print("\nAll downloads complete!")
