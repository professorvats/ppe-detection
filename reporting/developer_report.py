"""Developer Report Generator - Technical/Debug focused"""

from pathlib import Path
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from .data_processor import DetectionData
from .metrics_calculator import ModelMetrics


class DeveloperReportGenerator:
    """Generate comprehensive technical report for developers."""

    def __init__(self, config, output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, data: DetectionData, metrics: ModelMetrics) -> Path:
        """Generate complete developer PDF report."""
        output_path = self.output_dir / f'Developer_Report_{data.date_key}.pdf'
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("PPE Detection - Developer Report", styles['Title']))
        story.append(Paragraph(f"Date: {data.date_key}", styles['Normal']))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 20))

        # Metrics table
        story.append(Paragraph("Model Performance Metrics", styles['Heading2']))
        metrics_data = [
            ['Metric', 'Value'],
            ['Mean Confidence', f'{metrics.mean_confidence:.3f}'],
            ['Median Confidence', f'{metrics.median_confidence:.3f}'],
            ['Std Confidence', f'{metrics.std_confidence:.3f}'],
            ['Detection Rate', f'{metrics.detection_rate:.1%}'],
            ['Avg Detections/Image', f'{metrics.avg_detections_per_image:.1f}'],
            ['No Detection Count', str(metrics.no_detection_count)],
            ['Low Confidence Count', str(metrics.low_confidence_count)],
        ]
        table = Table(metrics_data, colWidths=[200, 150])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(table)
        story.append(Spacer(1, 20))

        # Class breakdown
        story.append(Paragraph("Detection Class Breakdown", styles['Heading2']))
        class_data = [['Class', 'Count', 'Avg Confidence']]
        for cls, count in sorted(metrics.class_counts.items(), key=lambda x: -x[1]):
            conf = metrics.class_confidence_means.get(cls, 0)
            class_data.append([cls, str(count), f'{conf:.3f}'])
        table = Table(class_data, colWidths=[150, 80, 100])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(table)

        doc.build(story)
        return output_path

    def export_supplementary_files(self, data: DetectionData) -> dict:
        """Export CSV, JSON, log files alongside PDF."""
        files = {}
        csv_path = self.output_dir / 'full_detections.csv'
        data.detections_df.to_csv(csv_path, index=False)
        files['full_csv'] = csv_path
        return files
