"""Metrics Calculator for PPE Detection Reports"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelMetrics:
    """Technical metrics for developer report"""
    mean_confidence: float = 0.0
    median_confidence: float = 0.0
    std_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0
    detection_rate: float = 0.0
    avg_detections_per_image: float = 0.0
    class_counts: Dict[str, int] = field(default_factory=dict)
    class_confidence_means: Dict[str, float] = field(default_factory=dict)
    no_detection_count: int = 0
    low_confidence_count: int = 0
    processing_errors: List[str] = field(default_factory=list)


@dataclass
class ComplianceMetrics:
    """Business metrics for client report"""
    overall_compliance_rate: float = 0.0
    helmet_compliance: float = 0.0
    vest_compliance: float = 0.0
    total_persons_detected: int = 0
    compliant_persons: int = 0
    status: str = 'unknown'  # 'green', 'yellow', 'red'


class MetricsCalculator:
    """Calculate technical and business metrics from detection data."""

    def __init__(self, config=None):
        self.config = config
        self.low_conf_threshold = 0.5

    def calculate_model_metrics(self, df: pd.DataFrame) -> ModelMetrics:
        """Calculate detailed technical metrics for developer report."""
        metrics = ModelMetrics()

        if df is None or len(df) == 0:
            return metrics

        actual_df = df[df['class'] != 'NO_DETECTION']
        conf_values = actual_df['confidence'].dropna()

        if len(conf_values) > 0:
            metrics.mean_confidence = float(conf_values.mean())
            metrics.median_confidence = float(conf_values.median())
            metrics.std_confidence = float(conf_values.std())
            metrics.min_confidence = float(conf_values.min())
            metrics.max_confidence = float(conf_values.max())

        total_images = df['image'].nunique()
        images_with_detections = actual_df['image'].nunique()
        metrics.detection_rate = images_with_detections / total_images if total_images > 0 else 0
        metrics.avg_detections_per_image = len(actual_df) / total_images if total_images > 0 else 0
        metrics.class_counts = actual_df['class'].value_counts().to_dict()
        metrics.no_detection_count = len(df[df['class'] == 'NO_DETECTION'])
        metrics.low_confidence_count = len(conf_values[conf_values < self.low_conf_threshold])

        for cls in actual_df['class'].unique():
            cls_conf = actual_df[actual_df['class'] == cls]['confidence'].dropna()
            if len(cls_conf) > 0:
                metrics.class_confidence_means[cls] = float(cls_conf.mean())

        return metrics

    def calculate_compliance_metrics(self, df: pd.DataFrame) -> ComplianceMetrics:
        """Calculate simplified compliance metrics for client report."""
        metrics = ComplianceMetrics()

        if df is None or len(df) == 0:
            return metrics

        actual_df = df[df['class'] != 'NO_DETECTION']
        person_count = len(actual_df[actual_df['class'].str.lower() == 'person'])
        metrics.total_persons_detected = person_count

        helmet_count = len(actual_df[actual_df['class'].isin(['Hardhat', 'helmet'])])
        vest_count = len(actual_df[actual_df['class'].isin(['Safety Vest', 'safety-vest'])])
        no_helmet = len(actual_df[actual_df['class'] == 'NO-Hardhat'])
        no_vest = len(actual_df[actual_df['class'] == 'NO-Safety Vest'])

        if person_count > 0:
            metrics.helmet_compliance = min(1.0, helmet_count / person_count)
            metrics.vest_compliance = min(1.0, vest_count / person_count)
            violation_rate = (no_helmet + no_vest) / (person_count * 2)
            metrics.overall_compliance_rate = max(0, 1 - violation_rate)
            metrics.compliant_persons = int(person_count * metrics.overall_compliance_rate)

        if metrics.overall_compliance_rate >= 0.90:
            metrics.status = 'green'
        elif metrics.overall_compliance_rate >= 0.70:
            metrics.status = 'yellow'
        else:
            metrics.status = 'red'

        return metrics
