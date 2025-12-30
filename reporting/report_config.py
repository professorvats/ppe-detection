"""
Configuration for PPE Reporting System
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict
from enum import Enum
import os


class ReportType(Enum):
    """Report type selection"""
    DEVELOPER = "developer"
    CLIENT = "client"
    BOTH = "both"


@dataclass
class ReportConfig:
    """Configuration for report generation"""

    # Paths
    base_dir: Path = field(default_factory=lambda: Path('.'))
    reports_dir: Path = field(default_factory=lambda: Path('reports'))

    # API keys
    openai_api_key: str = field(default_factory=lambda: os.getenv('OPENAI_API_KEY', ''))

    # Developer report settings
    include_all_images: bool = True
    include_raw_csv: bool = True
    low_confidence_threshold: float = 0.5

    # Client report settings
    max_sample_images: int = 6
    compliance_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'green': 0.90,   # 90%+ compliant
        'yellow': 0.70,  # 70-90% compliant
        'red': 0.0       # Below 70%
    })

    def __post_init__(self):
        self.base_dir = Path(self.base_dir)
        self.reports_dir = Path(self.reports_dir)


# PPE compliance definitions
REQUIRED_PPE = {'helmet', 'safety-vest', 'Hardhat', 'Safety Vest'}
OPTIONAL_PPE = {'gloves', 'glasses', 'safety-suit', 'Mask'}

# Class name mappings (for different model outputs)
PPE_CLASS_MAPPING = {
    'Hardhat': 'helmet',
    'Safety Vest': 'safety-vest',
    'NO-Hardhat': 'no-helmet',
    'NO-Safety Vest': 'no-safety-vest',
    'Person': 'person',
    'Mask': 'mask',
}

# Violation classes
VIOLATION_CLASSES = {'NO-Hardhat', 'NO-Safety Vest', 'NO-Mask', 'no-helmet', 'no-safety-vest'}
