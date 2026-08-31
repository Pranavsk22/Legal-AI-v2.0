import pytest
from pathlib import Path
from backend.nlp_modules.html_parser import extract_text_from_html
from backend.nlp_modules.universal_parser import extract_text, split_into_chunks

def test_html_parser(tmp_path):
    # Create a temporary HTML file
    html_file = tmp_path / "test.html"
    html_file.write_text("""
    <html>
        <head><title>Test Title</title></head>
        <body>
            <style>body { color: red; }</style>
            <script>console.log('hello');</script>
            <h1>Main Heading</h1>
            <p>This is a paragraph of the contract.</p>
            <div>Section 1.1: The Agreement</div>
        </body>
    </html>
    """, encoding="utf-8")
    
    extracted = extract_text_from_html(str(html_file))
    
    # Assert tag content is present
    assert "Main Heading" in extracted
    assert "This is a paragraph of the contract." in extracted
    assert "Section 1.1: The Agreement" in extracted
    
    # Assert scripts and styles are decomposed/stripped
    assert "console.log" not in extracted
    assert "color: red" not in extracted
    assert "Test Title" not in extracted

def test_universal_parser_txt(tmp_path):
    # Create a temporary text file
    txt_file = tmp_path / "test.txt"
    txt_content = "This is a raw text contract.\nSection 2: Term and Termination."
    txt_file.write_text(txt_content, encoding="utf-8")
    
    extracted = extract_text(str(txt_file))
    assert extracted == txt_content

def test_universal_parser_html(tmp_path):
    # Verify routing of HTML files in universal parser
    html_file = tmp_path / "test.html"
    html_file.write_text("<p>Hello HTML Universal Routing</p>", encoding="utf-8")
    
    extracted = extract_text(str(html_file))
    assert "Hello HTML Universal Routing" in extracted

def test_split_into_chunks():
    text = "Section 5 Liability is limited to $100. This is the first clause of the document. " \
           "We have some extra filler text here to ensure we get proper chunking. " \
           "Clause 9.1: The governing law is laws of California. This is another statement."
           
    chunks, metas = split_into_chunks(text, chunk_size=10, overlap=2)
    
    # We should have chunks and matching metadata
    assert len(chunks) > 0
    assert len(metas) == len(chunks)
    
    # Verify heading extraction inside chunk metas
    # HEADING_RE matches "section...", "clause...", "article..." at the beginning of window[:20]
    # For chunk 1 starting with "Section 5 Liability", it should trigger "Section 5"
    has_clause_5 = any("Section 5" in m["clause"] for m in metas)
    assert has_clause_5
