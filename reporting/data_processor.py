"""
Data Processor for PPE Detection Reports

Loads and processes detection data from result folders.
"""

import json
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class DetectionData:
    """Container for processed detection data"""
    summary: dict
    detections_df: pd.DataFrame
    date_key: str

    # Computed properties
    total_images: int = 0
    total_detections: int = 0
    unique_persons: int = 0
    compliance_rate: float = 0.0

    # Image paths
    annotated_images: List[Path] = field(default_factory=list)
    sample_images: List[Tuple[str, Path]] = field(default_factory=list)


class DataProcessor:
    """Loads and processes detection data from result folders."""

    def __init__(self, result_path: Path):
        """
        Initialize with path to result folder.

        Args:
            result_path: Path to result_X folder containing summary.json and CSVs
        """
        self.result_path = Path(result_path)
        self._summary = None
        self._df = None

    def load_data(self) -> Tuple[dict, pd.DataFrame]:
        """
        Load summary.json and all_detections.csv.

        Returns:
            Tuple of (summary dict, detections DataFrame)
        """
        # Load summary
        summary_path = self.result_path / 'summary.json'
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                self._summary = json.load(f)
        else:
            self._summary = {}

        # Load detections CSV
        csv_path = self.result_path / 'all_detections.csv'
        if csv_path.exists():
            self._df = pd.read_csv(csv_path)
        else:
            self._df = pd.DataFrame()

        return self._summary, self._df

    def get_dates(self) -> List[str]:
        """Get list of date keys from the data."""
        if self._summary is None:
            self.load_data()

        # Try to extract dates from camera names or folder structure
        cameras = self._summary.get('cameras', [])
        dates = set()

        for cam in cameras:
            # Parse date from camera name like "10-31_left"
            if '_' in str(cam):
                date_part = str(cam).split('_')[0]
                dates.add(date_part)

        if not dates and self._df is not None and len(self._df) > 0:
            # Try to extract from timestamps
            if 'timestamp' in self._df.columns:
                for ts in self._df['timestamp'].dropna().unique():
                    if 'T' in str(ts):
                        date_part = str(ts).split('T')[0]
                        dates.add(date_part)

        return sorted(list(dates)) if dates else ['all']

    def get_data_for_date(self, date_key: str = 'all') -> DetectionData:
        """
        Filter and process data for a specific date.

        Args:
            date_key: Date string (e.g., '10-31') or 'all'

        Returns:
            DetectionData object with filtered data
        """
        if self._summary is None or self._df is None:
            self.load_data()

        df = self._df.copy()

        # Filter by date if specified
        if date_key != 'all' and 'timestamp' in df.columns:
            df = df[df['timestamp'].str.contains(date_key, na=False)]

        # Calculate metrics
        actual_df = df[df['class'] != 'NO_DETECTION']
        person_df = actual_df[actual_df['class'].str.lower() == 'person']

        total_images = df['image'].nunique() if 'image' in df.columns else 0
        total_detections = len(actual_df)
        unique_persons = len(person_df)

        # Calculate compliance rate
        compliance_rate = self._calculate_compliance_rate(df)

        # Get image paths
        annotated_images = self.get_all_annotated_images(date_key)
        sample_images = self.get_sample_images(date_key)

        return DetectionData(
            summary=self._summary,
            detections_df=df,
            date_key=date_key,
            total_images=total_images,
            total_detections=total_detections,
            unique_persons=unique_persons,
            compliance_rate=compliance_rate,
            annotated_images=annotated_images,
            sample_images=sample_images
        )

    def _calculate_compliance_rate(self, df: pd.DataFrame) -> float:
        """Calculate PPE compliance percentage."""
        if df is None or len(df) == 0:
            return 0.0

        # Get images with persons
        person_images = df[df['class'].str.lower() == 'person']['image'].unique()
        if len(person_images) == 0:
            return 0.0

        # Check for violations
        violation_classes = {'NO-Hardhat', 'NO-Safety Vest', 'NO-Mask'}
        violation_images = df[df['class'].isin(violation_classes)]['image'].unique()

        compliant_images = len(person_images) - len(set(violation_images) & set(person_images))
        compliance_rate = compliant_images / len(person_images) if len(person_images) > 0 else 0.0

        return compliance_rate

    def get_all_annotated_images(self, date_key: str = 'all') -> List[Path]:
        """
        Get ALL annotated images for developer report.

        Args:
            date_key: Date filter

        Returns:
            List of paths to annotated images
        """
        images = []

        # Look in camera subdirectories
        for camera_dir in ['left', 'right', 'down']:
            camera_path = self.result_path / camera_dir
            if camera_path.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    images.extend(camera_path.glob(ext))

        # Also check date-prefixed directories
        for folder in self.result_path.iterdir():
            if folder.is_dir() and (date_key == 'all' or date_key in folder.name):
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    images.extend(folder.glob(ext))

        return sorted(images)

    def get_sample_images(
        self,
        date_key: str = 'all',
        num_per_camera: int = 2,
        quality_filter: str = 'best'
    ) -> List[Tuple[str, Path]]:
        """
        Get curated sample images for client report.

        Args:
            date_key: Date filter
            num_per_camera: Number of images per camera
            quality_filter: 'best', 'representative', or 'all'

        Returns:
            List of (camera_name, path) tuples
        """
        samples = []

        for camera in ['left', 'right', 'down']:
            camera_path = self.result_path / camera
            if not camera_path.exists():
                continue

            # Get images for this camera
            camera_images = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                camera_images.extend(camera_path.glob(ext))

            if not camera_images:
                continue

            # Sort by name (timestamp) and take samples
            camera_images = sorted(camera_images)

            if quality_filter == 'best':
                # Take first and last (different times)
                selected = camera_images[:num_per_camera]
            elif quality_filter == 'representative':
                # Take evenly spaced samples
                step = max(1, len(camera_images) // num_per_camera)
                selected = camera_images[::step][:num_per_camera]
            else:
                selected = camera_images[:num_per_camera]

            for img_path in selected:
                samples.append((camera, img_path))

        return samples

    def get_class_counts(self) -> Dict[str, int]:
        """Get detection counts by class."""
        if self._df is None:
            self.load_data()

        if self._df is None or len(self._df) == 0:
            return {}

        actual_df = self._df[self._df['class'] != 'NO_DETECTION']
        return actual_df['class'].value_counts().to_dict()

    def get_camera_stats(self) -> Dict[str, Dict]:
        """Get statistics per camera."""
        if self._df is None:
            self.load_data()

        if self._df is None or len(self._df) == 0:
            return {}

        stats = {}
        for camera in self._df['camera'].unique():
            cam_df = self._df[self._df['camera'] == camera]
            actual_cam_df = cam_df[cam_df['class'] != 'NO_DETECTION']

            stats[camera] = {
                'total_images': cam_df['image'].nunique(),
                'total_detections': len(actual_cam_df),
                'persons': len(actual_cam_df[actual_cam_df['class'].str.lower() == 'person']),
            }

        return stats

    def identify_edge_cases(self, low_conf_threshold: float = 0.5) -> pd.DataFrame:
        """
        Find low-confidence detections, failed detections, anomalies.

        Args:
            low_conf_threshold: Threshold for low confidence

        Returns:
            DataFrame of edge cases
        """
        if self._df is None:
            self.load_data()

        if self._df is None or len(self._df) == 0:
            return pd.DataFrame()

        edge_cases = []

        # No detections
        no_det = self._df[self._df['class'] == 'NO_DETECTION'].copy()
        no_det['issue'] = 'no_detection'
        edge_cases.append(no_det)

        # Low confidence
        actual_df = self._df[self._df['class'] != 'NO_DETECTION'].copy()
        low_conf = actual_df[actual_df['confidence'] < low_conf_threshold].copy()
        low_conf['issue'] = 'low_confidence'
        edge_cases.append(low_conf)

        # Load errors
        load_errors = self._df[self._df['class'] == 'LOAD_ERROR'].copy()
        load_errors['issue'] = 'load_error'
        edge_cases.append(load_errors)

        return pd.concat(edge_cases, ignore_index=True) if edge_cases else pd.DataFrame()
