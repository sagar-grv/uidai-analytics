

from xhtml2pdf import pisa
import markdown2
import os
import re

def convert_markdown_to_pdf(source_md_path, output_pdf_path):
    # 1. Read Markdown
    with open(source_md_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 2. Convert to HTML with tables and robust constraints
    html_content = markdown2.markdown(text, extras=["tables", "fenced-code-blocks", "cuddled-lists"])
    
    # 3. Add Strategic Framework Styling
    styled_html = f"""
    <html>
    <head>
        <style>
            @page {{ size: A4; margin: 2.5cm; }}
            body {{ font-family: 'Helvetica', sans-serif; font-size: 11pt; line-height: 1.5; color: #333; }}
            
            /* Hackathon Branding */
            h1 {{ color: #2E3E4E; font-size: 24pt; border-bottom: 3px solid #FF9933; padding-bottom: 10px; margin-top: 0; }}
            h2 {{ color: #138808; font-size: 18pt; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
            h3 {{ color: #E25555; font-size: 14pt; margin-top: 20px; }}
            
            /* Professional Tables */
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f8f9fa; color: #333; font-weight: bold; border-bottom: 2px solid #ddd; }}
            
            /* Code Blocks */
            pre {{ background-color: #f6f8fa; padding: 15px; border-radius: 6px; border: 1px solid #e1e4e8; overflow-x: auto; font-family: 'Courier New', monospace; font-size: 10pt; }}
            code {{ background-color: #f6f8fa; padding: 2px 4px; border-radius: 3px; font-family: 'Courier New', monospace; }}
            
            /* Images */
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; padding: 5px; margin: 15px 0; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }}
            
            .center {{ text-align: center; }}
        </style>
    </head>
    <body>
    {html_content}
    </body>
    </html>
    """

    # 4. Write to PDF
    print(f"Generating Report from: {source_md_path}")
    with open(output_pdf_path, "wb") as output_file:
        pisa_status = pisa.CreatePDF(styled_html, dest=output_file)

    if pisa_status.err:
        print(f"Error generating PDF: {pisa_status.err}")
    else:
        print(f"Successfully created PDF at: {output_pdf_path}")

if __name__ == "__main__":
    md_file = r"c:\Users\sagar\.gemini\antigravity\brain\f8544760-b88c-46f8-93e1-1110ec86efb7\walkthrough.md"
    pdf_file = "UIDAI_Project_Report.pdf"
    convert_markdown_to_pdf(md_file, pdf_file)

