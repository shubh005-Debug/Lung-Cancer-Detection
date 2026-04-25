from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from io import BytesIO
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Colors
BLUE = colors.HexColor("#2563eb")
BLUE_LIGHT = colors.HexColor("#eff6ff")
RED = colors.HexColor("#dc2626")
GREEN = colors.HexColor("#16a34a")
ORANGE = colors.HexColor("#ea580c")
GRAY_900 = colors.HexColor("#111827")
GRAY_600 = colors.HexColor("#4b5563")
GRAY_400 = colors.HexColor("#9ca3af")
GRAY_100 = colors.HexColor("#f3f4f6")
WHITE = colors.white

CLASS_LABELS = {
    "adenocarcinoma": "Adenocarcinoma",
    "large_cell_carcinoma": "Large Cell Carcinoma",
    "squamous_cell_carcinoma": "Squamous Cell Carcinoma",
    "normal": "Normal (No Cancer Detected)",
}


def generate_pdf_report(prediction_result, recommendations, patient_name="Anonymous"):
    """Generate a complete PDF report from prediction results."""
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20*mm,
        leftMargin=20*mm,
        topMargin=20*mm,
        bottomMargin=20*mm,
    )

    styles = getSampleStyleSheet()
    story = []

    # ── Custom Styles ──
    title_style = ParagraphStyle(
        "Title", parent=styles["Normal"],
        fontSize=22, textColor=GRAY_900,
        fontName="Helvetica-Bold", spaceAfter=10, leading=28,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        fontSize=12, textColor=GRAY_600,
        fontName="Helvetica-Bold", spaceAfter=8, leading=18,
    )
    section_header_style = ParagraphStyle(
        "SectionHeader", parent=styles["Normal"],
        fontSize=12, textColor=BLUE,
        fontName="Helvetica-Bold", spaceAfter=6, spaceBefore=14,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=9.5, textColor=GRAY_600,
        fontName="Helvetica", spaceAfter=4, leading=15,
    )
    bullet_style = ParagraphStyle(
        "Bullet", parent=styles["Normal"],
        fontSize=9.5, textColor=GRAY_600,
        fontName="Helvetica", spaceAfter=3, leading=14,
        leftIndent=12, bulletIndent=0,
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=styles["Normal"],
        fontSize=8, textColor=GRAY_400,
        fontName="Helvetica-Oblique", leading=12,
    )

    prediction = prediction_result.get("prediction", "unknown")
    confidence = prediction_result.get("confidence", 0)
    risk_level = prediction_result.get("risk_level", "Unknown")
    risk_color = RED if risk_level == "High" else GREEN
    probabilities = prediction_result.get("probabilities", {})
    report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    # ══════════════════════════════════════
    # HEADER
    # ══════════════════════════════════════
    header_data = [[
        Paragraph("LungAI", ParagraphStyle("Brand", parent=styles["Normal"],
            fontSize=16, textColor=BLUE, fontName="Helvetica-Bold")),
        Paragraph(f"<font color='#9ca3af'>Report generated: {report_date}</font>",
            ParagraphStyle("Date", parent=styles["Normal"],
            fontSize=8, fontName="Helvetica", alignment=2))
    ]]
    header_table = Table(header_data, colWidths=[90*mm, 80*mm])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=12))

    # Report Title
    story.append(Paragraph("Lung Cancer Detection Report", title_style))
    story.append(Paragraph(
        f'<font color="#9ca3af" size="9">PATIENT</font>',
        ParagraphStyle("PatientLabel", parent=styles["Normal"],
            fontSize=9, textColor=GRAY_400, fontName="Helvetica",
            spaceAfter=2, leading=14)
    ))
    story.append(Paragraph(f"{patient_name}", subtitle_style))
    story.append(Spacer(1, 10))

    # ══════════════════════════════════════
    # PREDICTION RESULT BOX
    # ══════════════════════════════════════
    story.append(Paragraph("AI Prediction Result", section_header_style))

    result_label = CLASS_LABELS.get(prediction, prediction.replace("_", " ").title())
    result_color = risk_color

    result_data = [
        [
            Paragraph(f"<b>Diagnosis</b>", ParagraphStyle("th", parent=styles["Normal"], fontSize=9, textColor=GRAY_400, fontName="Helvetica")),
            Paragraph(f"<b>Confidence</b>", ParagraphStyle("th", parent=styles["Normal"], fontSize=9, textColor=GRAY_400, fontName="Helvetica")),
            Paragraph(f"<b>Risk Level</b>", ParagraphStyle("th", parent=styles["Normal"], fontSize=9, textColor=GRAY_400, fontName="Helvetica")),
        ],
        [
            Paragraph(f"<b>{result_label}</b>", ParagraphStyle("val", parent=styles["Normal"], fontSize=13, textColor=GRAY_900, fontName="Helvetica-Bold")),
            Paragraph(f"<b>{confidence}%</b>", ParagraphStyle("val", parent=styles["Normal"], fontSize=13, textColor=BLUE, fontName="Helvetica-Bold")),
            Paragraph(f"<b>{risk_level}</b>", ParagraphStyle("val", parent=styles["Normal"], fontSize=13, textColor=result_color, fontName="Helvetica-Bold")),
        ],
    ]
    result_table = Table(result_data, colWidths=[80*mm, 45*mm, 45*mm])
    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), GRAY_100),
        ("BACKGROUND", (0,1), (-1,1), WHITE),
        ("BOX", (0,0), (-1,-1), 1, colors.HexColor("#e5e7eb")),
        ("INNERGRID", (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")),
        ("TOPPADDING", (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING", (0,0), (-1,-1), 12),
        ("ROWBACKGROUNDS", (0,1), (-1,1), [colors.HexColor("#fafafa")]),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 10))

    # ══════════════════════════════════════
    # PROBABILITY TABLE
    # ══════════════════════════════════════
    story.append(Paragraph("Class Probability Distribution", section_header_style))

    prob_rows = [[
        Paragraph("<b>Cancer Type</b>", ParagraphStyle("th", parent=styles["Normal"], fontSize=9, textColor=GRAY_400, fontName="Helvetica")),
        Paragraph("<b>Probability</b>", ParagraphStyle("th", parent=styles["Normal"], fontSize=9, textColor=GRAY_400, fontName="Helvetica")),
        Paragraph("<b>Percentage</b>", ParagraphStyle("th", parent=styles["Normal"], fontSize=9, textColor=GRAY_400, fontName="Helvetica")),
    ]]

    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    for cls, prob in sorted_probs:
        is_predicted = cls == prediction
        label = CLASS_LABELS.get(cls, cls.replace("_", " ").title())
        row_style = ParagraphStyle("row", parent=styles["Normal"],
            fontSize=9.5, fontName="Helvetica-Bold" if is_predicted else "Helvetica",
            textColor=BLUE if is_predicted else GRAY_600)
        prob_rows.append([
            Paragraph(f"{'→ ' if is_predicted else ''}{label}", row_style),
            Paragraph(f"{prob:.4f}", row_style),
            Paragraph(f"{prob*100:.1f}%", row_style),
        ])

    prob_table = Table(prob_rows, colWidths=[95*mm, 40*mm, 35*mm])
    prob_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), GRAY_100),
        ("BOX", (0,0), (-1,-1), 1, colors.HexColor("#e5e7eb")),
        ("INNERGRID", (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")),
        ("TOPPADDING", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING", (0,0), (-1,-1), 12),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, colors.HexColor("#fafafa")]),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 10))

    # ══════════════════════════════════════
    # RECOMMENDATIONS
    # ══════════════════════════════════════
    if recommendations:
        sections = [
            ("Precautions", "precautions"),
            ("Diet Recommendations", "diet_recommendations"),
            ("Lifestyle Advice", "lifestyle_advice"),
            ("Doctor Consultation", "doctor_consultation"),
        ]
        for title, key in sections:
            items = recommendations.get(key, [])
            if not items:
                continue
            story.append(Paragraph(title, section_header_style))
            for item in items:
                story.append(Paragraph(f"• {item}", bullet_style))
            story.append(Spacer(1, 4))

    # ══════════════════════════════════════
    # FOOTER DISCLAIMER
    # ══════════════════════════════════════
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY_400, spaceAfter=8))
    story.append(Paragraph(
        "Medical Disclaimer: This report is generated by an AI system for screening and research purposes only. "
        "It is NOT a substitute for professional medical diagnosis or treatment. "
        "Always consult a qualified physician or oncologist for medical advice and diagnosis.",
        disclaimer_style
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"Generated by LungAI Deep Learning System · VGG16 CNN · TensorFlow/Keras · {report_date}",
        ParagraphStyle("meta", parent=styles["Normal"], fontSize=7.5,
            textColor=GRAY_400, fontName="Helvetica", alignment=TA_CENTER)
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer