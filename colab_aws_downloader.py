#!/usr/bin/env python3
"""
=============================================================================
AWS S3 DOWNLOADER FOR PPE DETECTION RESEARCH
=============================================================================
Downloads images from S3 bucket for PPE detection training/testing.
Include this module in your Colab notebooks.
=============================================================================
"""

import os
import boto3
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================================
# AWS CONFIGURATION - SET YOUR CREDENTIALS HERE
# ============================================================================
AWS_CONFIG = {
    'access_key': os.environ.get('AWS_ACCESS_KEY_ID', ''),
    'secret_key': os.environ.get('AWS_SECRET_ACCESS_KEY', ''),
    'region': os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'),
    'bucket_name': 'your-ppe-bucket-name',  # Update with your bucket name
}

# ============================================================================
# S3 DOWNLOADER CLASS
# ============================================================================

class S3Downloader:
    """Download files from AWS S3 for PPE detection research"""

    def __init__(self, bucket_name: str = None, access_key: str = None, secret_key: str = None, region: str = 'us-east-1'):
        """Initialize S3 client"""
        self.bucket_name = bucket_name or AWS_CONFIG['bucket_name']
        self.access_key = access_key or AWS_CONFIG['access_key']
        self.secret_key = secret_key or AWS_CONFIG['secret_key']
        self.region = region or AWS_CONFIG['region']

        # Initialize boto3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region
        )

        print(f"  S3 Downloader initialized for bucket: {self.bucket_name}")

    def list_objects(self, prefix: str = '', max_keys: int = 1000) -> List[dict]:
        """List objects in S3 bucket with given prefix"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            return response.get('Contents', [])
        except Exception as e:
            print(f"  Error listing objects: {e}")
            return []

    def download_file(self, s3_key: str, local_path: str) -> bool:
        """Download a single file from S3"""
        try:
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            return True
        except Exception as e:
            print(f"  Error downloading {s3_key}: {e}")
            return False

    def download_folder(self, s3_prefix: str, local_dir: str, max_files: int = None,
                        file_extensions: List[str] = None, parallel: int = 4) -> int:
        """
        Download all files from S3 folder to local directory

        Args:
            s3_prefix: S3 folder prefix (e.g., 'images/2024-01-15/')
            local_dir: Local directory to save files
            max_files: Maximum number of files to download (None for all)
            file_extensions: List of extensions to filter (e.g., ['.jpg', '.png'])
            parallel: Number of parallel downloads

        Returns:
            Number of files downloaded
        """
        print(f"\n  Downloading from s3://{self.bucket_name}/{s3_prefix}")
        print(f"  To: {local_dir}")

        # List objects
        objects = self.list_objects(s3_prefix)

        if file_extensions:
            objects = [obj for obj in objects if any(obj['Key'].lower().endswith(ext) for ext in file_extensions)]

        if max_files:
            objects = objects[:max_files]

        if not objects:
            print("  No files found to download")
            return 0

        print(f"  Found {len(objects)} files to download")

        # Create local directory
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        # Download with progress
        downloaded = 0
        failed = 0

        def download_one(obj):
            s3_key = obj['Key']
            # Preserve folder structure
            relative_path = s3_key[len(s3_prefix):].lstrip('/')
            local_path = os.path.join(local_dir, relative_path)
            return self.download_file(s3_key, local_path)

        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {executor.submit(download_one, obj): obj for obj in objects}

            for i, future in enumerate(as_completed(futures), 1):
                if future.result():
                    downloaded += 1
                else:
                    failed += 1

                if i % 100 == 0 or i == len(objects):
                    print(f"  Progress: {i}/{len(objects)} ({downloaded} OK, {failed} failed)")

        print(f"  ✓ Downloaded {downloaded} files ({failed} failed)")
        return downloaded

    def download_date_folder(self, date_str: str, camera: str = 'down',
                             local_base: str = 's3_images', max_files: int = None) -> str:
        """
        Download images for a specific date and camera

        Args:
            date_str: Date string (e.g., '10-31', '11-7')
            camera: Camera name ('left', 'right', 'down')
            local_base: Base directory for downloads
            max_files: Max files to download

        Returns:
            Local path to downloaded images
        """
        s3_prefix = f"images/{date_str}/{camera}/"
        local_dir = os.path.join(local_base, date_str, camera)

        self.download_folder(
            s3_prefix=s3_prefix,
            local_dir=local_dir,
            max_files=max_files,
            file_extensions=['.jpg', '.jpeg', '.png']
        )

        return local_dir


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def setup_aws_credentials(access_key: str, secret_key: str, region: str = 'us-east-1'):
    """Set AWS credentials in environment"""
    os.environ['AWS_ACCESS_KEY_ID'] = access_key
    os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key
    os.environ['AWS_DEFAULT_REGION'] = region
    print("  ✓ AWS credentials configured")


def download_ppe_dataset(bucket_name: str, dates: List[str] = None,
                         cameras: List[str] = None, max_per_folder: int = None) -> str:
    """
    Download PPE dataset from S3

    Args:
        bucket_name: S3 bucket name
        dates: List of date folders to download (e.g., ['10-31', '11-7'])
        cameras: List of cameras (default: ['down'])
        max_per_folder: Max images per folder

    Returns:
        Path to downloaded dataset
    """
    dates = dates or ['10-31', '11-7']
    cameras = cameras or ['down']

    downloader = S3Downloader(bucket_name=bucket_name)

    for date in dates:
        for camera in cameras:
            print(f"\n  Downloading {date}/{camera}...")
            downloader.download_date_folder(date, camera, max_files=max_per_folder)

    return 's3_images'


# ============================================================================
# OPENAI INTEGRATION (Optional - for report generation)
# ============================================================================

def setup_openai(api_key: str):
    """Configure OpenAI API key"""
    os.environ['OPENAI_API_KEY'] = api_key
    print("  ✓ OpenAI API key configured")


def get_openai_analysis(prompt: str, model: str = 'gpt-4') -> Optional[str]:
    """Get analysis from OpenAI (optional enhancement for reports)"""
    try:
        import openai

        api_key = os.environ.get('OPENAI_API_KEY', '')
        if not api_key:
            return None

        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  OpenAI error: {e}")
        return None


# ============================================================================
# QUICK TEST
# ============================================================================

def test_connection():
    """Test AWS S3 connection"""
    print("\n" + "="*50)
    print("  TESTING AWS S3 CONNECTION")
    print("="*50)

    access_key = os.environ.get('AWS_ACCESS_KEY_ID', '')
    secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY', '')

    if not access_key or not secret_key:
        print("  ✗ AWS credentials not set!")
        print("  Run: setup_aws_credentials('YOUR_KEY', 'YOUR_SECRET')")
        return False

    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        # List buckets to test connection
        buckets = s3.list_buckets()
        print(f"  ✓ Connected! Found {len(buckets.get('Buckets', []))} buckets")
        return True
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        return False


if __name__ == '__main__':
    # Example usage:
    # setup_aws_credentials('YOUR_ACCESS_KEY', 'YOUR_SECRET_KEY')
    # test_connection()
    # download_ppe_dataset('your-bucket-name', dates=['10-31'], max_per_folder=100)
    pass
