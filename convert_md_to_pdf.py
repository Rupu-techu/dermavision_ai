import os
import sys
import subprocess
import markdown

def md_to_pdf(input_md, output_pdf):
    print(f"Reading markdown from: {input_md}")
    with open(input_md, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Convert markdown to HTML with tables and code highlighting
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "nl2br", "sane_lists"]
    )

    css = """
    @page {
        size: A4;
        margin: 18mm 16mm 18mm 16mm;
        @bottom-right {
            content: counter(page);
            font-size: 9pt;
            color: #64748b;
        }
    }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 10.5pt;
        line-height: 1.6;
        color: #1e293b;
        background: #ffffff;
        margin: 0;
        padding: 0;
    }
    h1 {
        font-size: 22pt;
        font-weight: 700;
        color: #0f172a;
        border-bottom: 2.5px solid #2563eb;
        padding-bottom: 8px;
        margin-top: 0;
        margin-bottom: 16px;
        page-break-after: avoid;
    }
    h2 {
        font-size: 15pt;
        font-weight: 700;
        color: #1e3a8a;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 6px;
        margin-top: 24px;
        margin-bottom: 12px;
        page-break-after: avoid;
    }
    h3 {
        font-size: 12pt;
        font-weight: 600;
        color: #1e293b;
        margin-top: 18px;
        margin-bottom: 8px;
        page-break-after: avoid;
    }
    p, li {
        color: #334155;
    }
    ul, ol {
        padding-left: 20px;
        margin-top: 6px;
        margin-bottom: 12px;
    }
    li {
        margin-bottom: 4px;
    }
    hr {
        border: none;
        border-top: 1px solid #cbd5e1;
        margin: 20px 0;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 14px 0;
        font-size: 9.5pt;
        page-break-inside: avoid;
    }
    th, td {
        border: 1px solid #cbd5e1;
        padding: 8px 10px;
        text-align: left;
    }
    th {
        background-color: #1e293b;
        color: #ffffff;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    tr:nth-child(even) td {
        background-color: #f8fafc;
    }
    code {
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
        font-size: 9pt;
        background-color: #f1f5f9;
        color: #0f172a;
        padding: 2px 5px;
        border-radius: 4px;
        border: 1px solid #e2e8f0;
    }
    pre {
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
        font-size: 8.5pt;
        background-color: #0f172a;
        color: #f8fafc;
        padding: 12px 14px;
        border-radius: 6px;
        overflow-x: auto;
        line-height: 1.45;
        margin: 12px 0;
        page-break-inside: avoid;
    }
    pre code {
        background-color: transparent;
        color: inherit;
        padding: 0;
        border: none;
        font-size: inherit;
    }
    blockquote {
        margin: 12px 0;
        padding: 10px 14px;
        background-color: #f0fdf4;
        border-left: 4px solid #16a34a;
        color: #166534;
        border-radius: 0 4px 4px 0;
        page-break-inside: avoid;
    }
    blockquote p {
        margin: 0;
        color: inherit;
    }
    a {
        color: #2563eb;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    """

    html_full = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>DermaVision AI — Project Audit & Module 5 Progress</title>
    <style>
    {css}
    </style>
</head>
<body>
    {html_body}
</body>
</html>"""

    tmp_html = input_md.replace(".md", ".html")
    with open(tmp_html, "w", encoding="utf-8") as f:
        f.write(html_full)
    print(f"Generated intermediate HTML at: {tmp_html}")

    chrome_cmd = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={output_pdf}",
        "--no-pdf-header-footer",
        tmp_html
    ]

    print("Running Headless Chrome to generate PDF...")
    result = subprocess.run(chrome_cmd, capture_output=True, text=True)
    if result.returncode == 0 and os.path.exists(output_pdf):
        print(f"SUCCESS: Created PDF at: {output_pdf}")
        print(f"File size: {os.path.getsize(output_pdf) / 1024:.2f} KB")
        # Clean up temporary HTML
        if os.path.exists(tmp_html):
            os.remove(tmp_html)
        return output_pdf
    else:
        print(f"ERROR: Return code {result.returncode}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return None

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.dirname(__file__))
    input_file = os.path.join(base_dir, "PROJECT_AUDIT_AND_CHAT_SUMMARY.md")
    output_file = os.path.join(base_dir, "PROJECT_AUDIT_AND_CHAT_SUMMARY.pdf")
    md_to_pdf(input_file, output_file)
