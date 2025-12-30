"""Client Report Generator - Business focused, no technical jargon"""

from pathlib import Path
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from .data_processor import DetectionData
from .metrics_calculator import ComplianceMetrics


class ClientReportGenerator:
    """Generate clean, professional client-facing report."""

    def __init__(self, config, output_dir: Path):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, data: DetectionData, metrics: ComplianceMetrics) -> Path:
        """Generate complete client PDF report."""
        output_path = self.output_dir / f'PPE_Compliance_Report_{data.date_key}.pdf'
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Title
        story.append(Paragraph("Workplace Safety Compliance Report", styles['Title']))
        story.append(Paragraph(f"Report Date: {data.date_key}", styles['Normal']))
        story.append(Spacer(1, 30))

        # Status
        status_color = {'green': colors.green, 'yellow': colors.orange, 'red': colors.red}.get(metrics.status, colors.grey)
        story.append(Paragraph(f"Overall Status: {metrics.status.upper()}", styles['Heading1']))
        story.append(Paragraph(f"Compliance Rate: {metrics.overall_compliance_rate:.0%}", styles['Heading2']))
        story.append(Spacer(1, 20))

        # Key stats
        story.append(Paragraph("Key Statistics", styles['Heading2']))
        stats_data = [
            ['Metric', 'Value'],
            ['Workers Monitored', str(metrics.total_persons_detected)],
            ['Compliant Workers', str(metrics.compliant_persons)],
            ['Helmet Compliance', f'{metrics.helmet_compliance:.0%}'],
            ['Vest Compliance', f'{metrics.vest_compliance:.0%}'],
        ]
        table = Table(stats_data, colWidths=[200, 150])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2196F3')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
        ]))
        story.append(table)
        story.append(Spacer(1, 30))

        # Recommendations
        story.append(Paragraph("Recommendations", styles['Heading2']))
        if metrics.status == 'green':
            story.append(Paragraph("Excellent compliance! Continue current safety practices.", styles['Normal']))
        elif metrics.status == 'yellow':
            story.append(Paragraph("Room for improvement. Consider additional safety training.", styles['Normal']))
        else:
            story.append(Paragraph("Immediate attention required. Schedule safety review.", styles['Normal']))

        doc.build(story)
        return output_path
