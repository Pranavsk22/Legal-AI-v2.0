from bs4 import BeautifulSoup

def extract_text_from_html(path: str) -> str:
    """Extract clean text content from an HTML file, stripping script and style elements."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, "html.parser")
    
    # decompose script and style elements
    for element in soup(["script", "style", "head", "title", "meta"]):
        element.decompose()
        
    # get text content with newline separator
    text = soup.get_text(separator="\n")
    
    # Clean up empty lines / spacing
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)
