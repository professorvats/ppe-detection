#!/usr/bin/env python3
"""
PPE Detection Report Generator - Main Entry Point

Generates Developer and/or Client reports from detection results.

Usage:
    python generate_reports.py                    # Interactive mode
    python generate_reports.py --type developer   # Developer report only
    python generate_reports.py --type client      # Client report only
    python generate_reports.py --type both        # Both reports
    python generate_reports.py --folder result_1  # Specify result folder
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

from reporting import (
    ReportConfig,
    ReportType,
    DataProcessor,
    MetricsCalculator,
)
from reporting.developer_report import DeveloperReportGenerator
from reporting.client_report import ClientReportGenerator


def find_result_folders(base_dir: Path = Path('.')) -> list:
    """Find all valid result_* folders."""
    folders = []
    for p in sorted(base_dir.glob('result_*')):
        if p.is_dir() and (p / 'summary.json').exists():
            folders.append(p)
    return folders


def main():
    parser = argparse.ArgumentParser(description='Generate PPE Detection Reports')
    parser.add_argument('--type', choices=['developer', 'client', 'both'],
                        default='both', help='Report type to generate')
    parser.add_argument('--folder', type=str, default=None,
                        help='Result folder name (e.g., result_1)')
    args = parser.parse_args()

    print("=" * 70)
    print("  PPE DETECTION REPORT GENERATOR")
    print("=" * 70)

    # Find result folders
    result_folders = find_result_folders()
    if not result_folders:
        print("\n  No result folders found!")
        print("  Run ppe_report.py first to generate detection results.")
        sys.exit(1)

    # Select folder
    if args.folder:
        result_folder = Path(args.folder)
    else:
        result_folder = result_folders[-1]  # Use latest

    print(f"\n  Using: {result_folder}")

    # Initialize
    config = ReportConfig()
    processor = DataProcessor(result_folder)
    summary, df = processor.load_data()
    metrics_calc = MetricsCalculator(config)

    # Generate reports
    report_type = ReportType(args.type)
    reports_dir = Path('reports') / result_folder.name

    for date_key in processor.get_dates():
        data = processor.get_data_for_date(date_key)
        print(f"\n  Processing: {date_key}")

        if report_type in (ReportType.DEVELOPER, ReportType.BOTH):
            model_metrics = metrics_calc.calculate_model_metrics(data.detections_df)
            dev_output = reports_dir / date_key / "developer"
            dev_gen = DeveloperReportGenerator(config, dev_output)
            dev_report = dev_gen.generate(data, model_metrics)
            dev_gen.export_supplementary_files(data)
            print(f"    Developer report: {dev_report}")

        if report_type in (ReportType.CLIENT, ReportType.BOTH):
            compliance_metrics = metrics_calc.calculate_compliance_metrics(data.detections_df)
            client_output = reports_dir / date_key / "client"
            client_gen = ClientReportGenerator(config, client_output)
            client_report = client_gen.generate(data, compliance_metrics)
            print(f"    Client report: {client_report}")

    print("\n" + "=" * 70)
    print("  REPORTS GENERATED!")
    print("=" * 70)


if __name__ == '__main__':
    main()
