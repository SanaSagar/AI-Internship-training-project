from fpdf import FPDF
from datetime import datetime

class ReportPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        # Left-aligned title
        self.cell(0, 10, 'LLM Prompt Evaluator', 0, 0, 'L')
        # Right-aligned timestamp
        self.set_xy(0, self.get_y())
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.set_font('Helvetica', '', 10)
        self.cell(0, 10, timestamp_str, 0, 1, 'R')
        # Thin horizontal rule
        self.set_draw_color(200, 200, 200) # Light gray line
        self.set_line_width(0.2)
        self.line(self.get_x(), self.get_y(), 210 - self.get_x(), self.get_y())
        self.ln(5)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        # Centered page number
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


def generate_pdf_report(prompt, expected_output, model_output, final_score, hallucination_result):
    """
    Generate a simple, structured 5-section PDF report using FPDF2.
    """
    pdf = ReportPDF(orientation='P', unit='mm', format='A4')
    pdf.set_margins(left=20, top=20, right=20)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    
    def safe_text(text):
        if not text:
            return "None provided"
        # Since we use default Helvetica font, we must encode to latin-1 replacing unknown chars
        return str(text).encode('latin-1', 'replace').decode('latin-1')
        
    def add_section(label, content, is_last=False):
        # Section Label
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(51, 51, 51) # #333333 Dark Gray
        pdf.cell(0, 8, safe_text(label), ln=True)
        
        # Section Content
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(0, 0, 0) # Black
        pdf.multi_cell(0, 6, safe_text(content))
        
        if not is_last:
            pdf.ln(3) # Pad before line
            # Divider line
            pdf.set_draw_color(204, 204, 204) # #CCCCCC Light Gray
            pdf.set_line_width(0.2)
            y = pdf.get_y()
            pdf.line(20, y, 190, y)
            pdf.ln(6) # 6mm spacing

    # 1. Prompt
    add_section("Prompt", prompt)

    # 2. Expected Output
    add_section("Expected Output", expected_output)

    # 3. Model Output
    add_section("Model Output", model_output)

    # 4. Final Score
    add_section("Final Score", f"{final_score:.1f} / 100")

    # 5. Hallucination Result
    # Custom rendering for Hallucination Result due to color requirements
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(51, 51, 51)
    pdf.cell(0, 8, "Hallucination Risk", ln=True)

    risk_level = "UNKNOWN"
    rules = []
    if hallucination_result:
        risk_level = hallucination_result.get("risk_level", "UNKNOWN").upper()
        rules = hallucination_result.get("triggered_rules", [])
        
    # Risk Level
    pdf.set_font('Helvetica', 'B', 10)
    if risk_level == "HIGH":
        pdf.set_text_color(204, 0, 0) # #CC0000
    elif risk_level == "MEDIUM":
        pdf.set_text_color(232, 119, 34) # #E87722
    elif risk_level == "LOW":
        pdf.set_text_color(16, 185, 129) # #10B981
    else:
        pdf.set_text_color(0, 0, 0)
        
    pdf.cell(0, 6, risk_level, ln=True)

    # Triggered Rules
    if rules:
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(128, 128, 128) # Gray for rules
        for rule in rules:
            pdf.multi_cell(0, 6, safe_text(f"  • {rule}"))

    return bytes(pdf.output())
