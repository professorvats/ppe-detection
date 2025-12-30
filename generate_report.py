#!/usr/bin/env python3
"""
PPE Detection Report Generator
Auto-detects result folders, generates separate PDF reports per date.
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

# Configuration
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = BASE_DIR / "reports"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Set via environment variable


def find_result_folders():
    """Find all result_* folders with valid data."""
    results = []

    for folder in sorted(BASE_DIR.glob("result_*")):
        if not folder.is_dir():
            continue

        summary_path = folder / "summary.json"
        csv_path = folder / "all_detections.csv"

        if summary_path.exists() and csv_path.exists():
            try:
                with open(summary_path, 'r') as f:
                    summary = json.load(f)

                # Get folder stats
                num_cameras = len(summary.get('cameras', []))
                total_images = summary.get('total_images', 0)
                total_detections = summary.get('total_detections', 0)
                generated = summary.get('generated', 'Unknown')

                # Extract unique dates from camera names
                dates = set()
                for cam in summary.get('cameras', []):
                    date_part = cam.split('_')[0]
                    dates.add(date_part)

                # Parse timestamp
                try:
                    dt = datetime.fromisoformat(generated)
                    date_str = dt.strftime('%Y-%m-%d %H:%M')
                except:
                    date_str = generated[:16] if len(generated) >= 16 else generated

                results.append({
                    'path': folder,
                    'name': folder.name,
                    'summary': summary,
                    'cameras': num_cameras,
                    'images': total_images,
                    'detections': total_detections,
                    'date': date_str,
                    'dates': sorted(dates)
                })
            except Exception as e:
                print(f"  Warning: Could not read {folder.name}: {e}")

    return results


def display_results_menu(results):
    """Display available result folders and get user selection."""
    print("\n" + "=" * 70)
    print("  AVAILABLE RESULT FOLDERS")
    print("=" * 70)

    if not results:
        print("\n  No result folders found!")
        print("  Run the PPE training notebook first to generate results.")
        return None

    print(f"\n  {'#':<4} {'Folder':<15} {'Date':<18} {'Days':<12} {'Images':<10} {'Detections':<12}")
    print(f"  {'-'*4} {'-'*15} {'-'*18} {'-'*12} {'-'*10} {'-'*12}")

    for i, r in enumerate(results, 1):
        days_str = ', '.join(r['dates'])
        print(f"  {i:<4} {r['name']:<15} {r['date']:<18} {days_str:<12} {r['images']:<10,} {r['detections']:<12,}")

    print(f"\n  {'-'*70}")
    print(f"  Enter number to select (1-{len(results)}), or 'q' to quit")
    print(f"  {'-'*70}")

    while True:
        try:
            choice = input("\n  Your choice: ").strip().lower()
            if choice == 'q':
                return None

            num = int(choice)
            if 1 <= num <= len(results):
                return results[num - 1]
            else:
                print(f"  Please enter a number between 1 and {len(results)}")
        except ValueError:
            print("  Invalid input. Enter a number or 'q' to quit.")


def load_data(result_folder):
    """Load detection data from result folder."""
    folder_path = result_folder['path']

    # Load summary
    with open(folder_path / "summary.json", "r") as f:
        summary = json.load(f)

    # Load all detections
    df = pd.read_csv(folder_path / "all_detections.csv")

    return summary, df


def get_data_for_date(summary, df, date_key):
    """Filter summary and dataframe for a specific date."""
    # Filter cameras for this date
    cameras = [c for c in summary.get('cameras', []) if c.startswith(date_key)]

    # Filter dataframe
    df_date = df[df['camera'].str.startswith(date_key)]

    # Build date-specific summary
    date_summary = {
        'generated': summary.get('generated'),
        'result_folder': summary.get('result_folder'),
        'model': summary.get('model'),
        'time_filter': summary.get('time_filter'),
        'date': date_key,
        'cameras': cameras,
        'images_per_camera': {k: v for k, v in summary.get('images_per_camera', {}).items() if k.startswith(date_key)},
        'detections_per_camera': {k: v for k, v in summary.get('detections_per_camera', {}).items() if k.startswith(date_key)},
        'total_images': sum(v for k, v in summary.get('images_per_camera', {}).items() if k.startswith(date_key)),
        'total_detections': sum(v for k, v in summary.get('detections_per_camera', {}).items() if k.startswith(date_key)),
    }

    # Calculate by_class for this date
    actual_det = df_date[df_date['class'] != 'NO_DETECTION']
    date_summary['by_class'] = actual_det['class'].value_counts().to_dict()

    return date_summary, df_date


def generate_charts(summary, df, output_dir):
    """Generate visualization charts."""
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    chart_paths = []

    # 1. Detection by Class (Bar Chart)
    by_class = summary.get("by_class", {})
    if by_class:
        plt.figure(figsize=(10, 6))
        classes = list(by_class.keys())
        counts = list(by_class.values())
        colors_list = plt.cm.viridis([i/max(len(classes), 1) for i in range(len(classes))])
        bars = plt.bar(classes, counts, color=colors_list)
        plt.xlabel("Detection Class", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.title(f"PPE Detections by Class - {summary.get('date', '')}", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                     f"{count:,}", ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        chart1_path = charts_dir / "detections_by_class.png"
        plt.savefig(chart1_path, dpi=150, bbox_inches='tight')
        plt.close()
        chart_paths.append(chart1_path)

    # 2. Confidence Distribution (Histogram)
    df_with_conf = df[df['confidence'].notna()]
    if len(df_with_conf) > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(df_with_conf['confidence'], bins=20, color='#2ecc71', edgecolor='black', alpha=0.7)
        plt.xlabel("Confidence Score", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title(f"Confidence Distribution - {summary.get('date', '')}", fontsize=14, fontweight='bold')
        mean_conf = df_with_conf['confidence'].mean()
        plt.axvline(mean_conf, color='red', linestyle='--', label=f'Mean: {mean_conf:.3f}')
        plt.legend()
        plt.tight_layout()
        chart2_path = charts_dir / "confidence_distribution.png"
        plt.savefig(chart2_path, dpi=150, bbox_inches='tight')
        plt.close()
        chart_paths.append(chart2_path)

    # 3. Detections by Camera (Bar Chart)
    det_by_cam = summary.get('detections_per_camera', {})
    if det_by_cam:
        plt.figure(figsize=(10, 6))
        cameras = list(det_by_cam.keys())
        counts = list(det_by_cam.values())
        # Shorten camera names for display (remove date prefix)
        short_names = [c.split('_', 1)[1] if '_' in c else c for c in cameras]
        colors_list = plt.cm.tab10([i/max(len(cameras), 1) for i in range(len(cameras))])
        bars = plt.bar(short_names, counts, color=colors_list)
        plt.xlabel("Camera", fontsize=12)
        plt.ylabel("Detections", fontsize=12)
        plt.title(f"Detections by Camera - {summary.get('date', '')}", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                     f"{count:,}", ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        chart3_path = charts_dir / "detections_by_camera.png"
        plt.savefig(chart3_path, dpi=150, bbox_inches='tight')
        plt.close()
        chart_paths.append(chart3_path)

    return chart_paths


def get_chatgpt_analysis(summary, df, api_key):
    """Send data to ChatGPT for analysis."""
    client = OpenAI(api_key=api_key)

    # Prepare data summary
    df_with_conf = df[df['confidence'].notna()]

    # Build camera stats
    camera_stats = ""
    images_per_cam = summary.get('images_per_camera', {})
    det_per_cam = summary.get('detections_per_camera', {})
    for cam in summary.get('cameras', []):
        imgs = images_per_cam.get(cam, 0)
        dets = det_per_cam.get(cam, 0)
        camera_stats += f"    - {cam}: {imgs:,} images, {dets:,} detections\n"

    # Safe confidence stats
    if len(df_with_conf) > 0:
        mean_conf = f"{df_with_conf['confidence'].mean():.4f}"
        median_conf = f"{df_with_conf['confidence'].median():.4f}"
        min_conf = f"{df_with_conf['confidence'].min():.4f}"
        max_conf = f"{df_with_conf['confidence'].max():.4f}"
        std_conf = f"{df_with_conf['confidence'].std():.4f}"
    else:
        mean_conf = median_conf = min_conf = max_conf = std_conf = "N/A"

    data_summary = f"""
    PPE Detection Analysis Data for Date: {summary.get('date', 'Unknown')}

    Model: {summary.get('model', 'Unknown')}
    Time Filter: {summary.get('time_filter', 'Unknown')}
    Generated: {summary.get('generated', 'Unknown')}

    CAMERA STATISTICS:
{camera_stats}
    Total images processed: {summary.get('total_images', 0):,}
    Total detections: {summary.get('total_detections', 0):,}

    DETECTIONS BY CLASS:
    {json.dumps(summary.get('by_class', {}), indent=2)}

    CONFIDENCE STATISTICS:
    - Mean confidence: {mean_conf}
    - Median confidence: {median_conf}
    - Min confidence: {min_conf}
    - Max confidence: {max_conf}
    - Std deviation: {std_conf}

    NO_DETECTION count: {len(df[df['class'] == 'NO_DETECTION']):,}
    Images with detections: {len(df[df['class'] != 'NO_DETECTION']['image'].unique()):,}
    """

    prompt = f"""You are a workplace safety analyst. Analyze the following PPE (Personal Protective Equipment) detection data from a computer vision system monitoring a workplace on {summary.get('date', 'a specific date')}.

{data_summary}

Please provide a comprehensive analysis report including:

1. EXECUTIVE SUMMARY (2-3 paragraphs)
   - Key findings and overall assessment for this date
   - Detection system performance overview

2. PPE COMPLIANCE ANALYSIS
   - Analysis of detected PPE items (helmets, vests, etc.)
   - Identification of potential compliance gaps
   - Areas of concern

3. CAMERA PERFORMANCE COMPARISON
   - Compare camera effectiveness (left, right, down)
   - Detection rate analysis by camera position

4. CONFIDENCE SCORE ANALYSIS
   - Assessment of detection reliability
   - Recommendations for threshold settings

5. KEY INSIGHTS & PATTERNS
   - Notable patterns in the data
   - Observations about worker behavior

6. RECOMMENDATIONS
   - Actionable recommendations for improving workplace safety
   - Suggestions for improving detection system

7. CONCLUSION
   - Summary of critical points
   - Priority areas for attention

Format the response in clear sections with headers. Be specific and data-driven in your analysis."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert workplace safety analyst specializing in PPE compliance and computer vision systems."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4000,
        temperature=0.7
    )

    return response.choices[0].message.content


def get_sample_images(result_path, date_key, num_per_camera=2):
    """Get sample detection images from camera folders for a specific date."""
    images = []

    # Find camera subfolders for this date
    for folder in sorted(result_path.iterdir()):
        if folder.is_dir() and folder.name.startswith(date_key):
            jpg_files = list(folder.glob('*.jpg'))[:num_per_camera]
            for img in jpg_files:
                # Use short camera name
                short_name = folder.name.split('_', 1)[1] if '_' in folder.name else folder.name
                images.append((short_name, str(img)))

    return images[:12]  # Max 12 images total


def create_pdf_report(summary, analysis, chart_paths, sample_images, output_path):
    """Generate PDF report with analysis, charts, and images."""
    doc = SimpleDocTemplate(str(output_path), pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=72)

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#2c3e50')
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#2980b9')
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        leading=14
    )

    story = []

    # Title Page
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("PPE Detection Analysis Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(f"Date: {summary.get('date', 'Unknown')}",
                          ParagraphStyle('Center', parent=styles['Normal'], alignment=TA_CENTER, fontSize=18, textColor=colors.HexColor('#e74c3c'))))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}",
                          ParagraphStyle('Center', parent=styles['Normal'], alignment=TA_CENTER)))
    story.append(Spacer(1, 0.25*inch))
    story.append(Paragraph(f"Model: {summary.get('model', 'Unknown')}",
                          ParagraphStyle('Center', parent=styles['Normal'], alignment=TA_CENTER)))
    story.append(Paragraph(f"Time Filter: {summary.get('time_filter', 'Unknown')}",
                          ParagraphStyle('Center', parent=styles['Normal'], alignment=TA_CENTER)))
    story.append(PageBreak())

    # Summary Statistics Table
    story.append(Paragraph("Detection Summary", heading_style))

    summary_data = [
        ["Metric", "Value"],
        ["Date", summary.get('date', 'Unknown')],
        ["Total Images Processed", f"{summary.get('total_images', 0):,}"],
        ["Total Detections", f"{summary.get('total_detections', 0):,}"],
        ["Number of Cameras", f"{len(summary.get('cameras', []))}"],
    ]

    # Add camera breakdown
    for cam in summary.get('cameras', []):
        short_name = cam.split('_', 1)[1] if '_' in cam else cam
        imgs = summary.get('images_per_camera', {}).get(cam, 0)
        dets = summary.get('detections_per_camera', {}).get(cam, 0)
        summary_data.append([f"  {short_name}", f"{imgs:,} imgs / {dets:,} dets"])

    table = Table(summary_data, colWidths=[3*inch, 2.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.5*inch))

    # Class Breakdown Table
    by_class = summary.get('by_class', {})
    if by_class:
        story.append(Paragraph("Detection Classes", heading_style))
        class_data = [["Class", "Count", "Percentage"]]
        total = sum(by_class.values())
        for cls, count in sorted(by_class.items(), key=lambda x: -x[1]):
            pct = count / total * 100 if total > 0 else 0
            class_data.append([cls, f"{count:,}", f"{pct:.1f}%"])

        class_table = Table(class_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        class_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ecf0f1')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(class_table)

    story.append(PageBreak())

    # ChatGPT Analysis
    story.append(Paragraph("AI-Powered Analysis", title_style))
    story.append(Spacer(1, 0.25*inch))

    # Parse and format the analysis
    for line in analysis.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Check if it's a header
        if line.startswith('#') or '**' in line:
            clean_line = line.replace('#', '').replace('**', '').strip()
            story.append(Paragraph(clean_line, heading_style))
        elif line[0:1].isdigit() and '.' in line[:3]:
            # Numbered section header
            story.append(Paragraph(line, heading_style))
        else:
            # Regular paragraph - escape special characters
            safe_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(safe_line, body_style))

    story.append(PageBreak())

    # Charts Section
    story.append(Paragraph("Visual Analytics", title_style))
    story.append(Spacer(1, 0.25*inch))

    for i, chart_path in enumerate(chart_paths):
        if chart_path.exists():
            img = Image(str(chart_path), width=6*inch, height=3.5*inch)
            story.append(img)
            story.append(Spacer(1, 0.3*inch))

            if (i + 1) % 2 == 0 and i < len(chart_paths) - 1:
                story.append(PageBreak())

    story.append(PageBreak())

    # Sample Images Section
    if sample_images:
        story.append(Paragraph("Sample Detection Images", title_style))
        story.append(Spacer(1, 0.25*inch))

        for label, img_path in sample_images:
            if os.path.exists(img_path):
                story.append(Paragraph(f"{label}: {os.path.basename(img_path)}",
                                      ParagraphStyle('ImageLabel', parent=styles['Normal'],
                                                    fontSize=9, textColor=colors.grey)))
                img = Image(img_path, width=5*inch, height=3.5*inch)
                story.append(img)
                story.append(Spacer(1, 0.3*inch))

    # Build PDF
    doc.build(story)
    return output_path


def main():
    """Main function to generate reports - one per date."""
    print("=" * 70)
    print("  PPE Detection Report Generator")
    print("=" * 70)

    # Find available result folders
    results = find_result_folders()

    # Check for command-line argument (auto-select)
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.isdigit():
            num = int(arg)
            if 1 <= num <= len(results):
                selected = results[num - 1]
                print(f"\n  Auto-selected: {selected['name']}")
            else:
                print(f"\n  Invalid selection: {num}. Available: 1-{len(results)}")
                return
        else:
            # Try to match folder name
            matches = [r for r in results if arg in r['name']]
            if matches:
                selected = matches[0]
                print(f"\n  Auto-selected: {selected['name']}")
            else:
                print(f"\n  No folder matching '{arg}' found.")
                return
    else:
        # Display menu and get selection
        selected = display_results_menu(results)

    if selected is None:
        print("\nNo folder selected. Exiting.")
        return

    print(f"\n  Selected: {selected['name']}")
    print(f"  Dates found: {', '.join(selected['dates'])}")

    # Use configured API key
    api_key = OPENAI_API_KEY

    # Load full data
    print(f"\nLoading data from {selected['name']}...")
    summary, df = load_data(selected)

    # Generate report for each date
    generated_reports = []

    for date_key in selected['dates']:
        print(f"\n{'='*70}")
        print(f"  GENERATING REPORT FOR: {date_key}")
        print(f"{'='*70}")

        # Create date-specific output folder
        date_output_dir = REPORTS_DIR / selected['name'] / date_key
        date_output_dir.mkdir(parents=True, exist_ok=True)

        # Get date-specific data
        date_summary, date_df = get_data_for_date(summary, df, date_key)

        print(f"  Images: {date_summary['total_images']:,}")
        print(f"  Detections: {date_summary['total_detections']:,}")
        print(f"  Cameras: {', '.join(date_summary['cameras'])}")

        # Generate charts
        print(f"  Generating charts...")
        chart_paths = generate_charts(date_summary, date_df, date_output_dir)

        # Get ChatGPT analysis
        print(f"  Getting AI analysis...")
        try:
            analysis = get_chatgpt_analysis(date_summary, date_df, api_key)
        except Exception as e:
            print(f"  Warning: ChatGPT analysis failed: {e}")
            analysis = "AI analysis not available."

        # Get sample images
        sample_images = get_sample_images(selected['path'], date_key)

        # Create PDF report
        print(f"  Creating PDF report...")
        pdf_path = date_output_dir / f"PPE_Report_{date_key}.pdf"
        create_pdf_report(date_summary, analysis, chart_paths, sample_images, pdf_path)

        generated_reports.append(pdf_path)
        print(f"  Saved: {pdf_path}")

    # Summary
    print("\n" + "=" * 70)
    print("  REPORT GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\n  Generated {len(generated_reports)} report(s):\n")
    for report in generated_reports:
        print(f"    {report}")
    print("\n" + "=" * 70)

    # Open reports folder
    os.system(f'open "{REPORTS_DIR / selected["name"]}"')


if __name__ == "__main__":
    main()
