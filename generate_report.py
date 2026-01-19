
from xhtml2pdf import pisa
import markdown2
import os

def convert_markdown_to_pdf(source_md_path, output_pdf_path):
    # 1. Read Markdown
    with open(source_md_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 2. Convert to HTML with basic alignment styles
    html_content = markdown2.markdown(text, extras=["tables", "fenced-code-blocks"])
    
    # 3. Add styling for report
    styled_html = f"""
    <html>
    <head>
        <style>
            @page {{ size: A4; margin: 2cm; }}
            body {{ font-family: Helvetica, sans-serif; font-size: 11pt; }}
            h1 {{ color: #2E3E4E; border-bottom: 2px solid #2E3E4E; padding-bottom: 5px; }}
            h2 {{ color: #E25555; margin-top: 20px; }}
            h3 {{ color: #138808; margin-top: 15px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px; font-family: monospace; }}
            pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        </style>
    </head>
    <body>
    {html_content}
    </body>
    </html>
    """

    # 4. Write to PDF
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
