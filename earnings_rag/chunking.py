from pathlib import Path
from bs4 import BeautifulSoup

DROP_TAGS = ['script', 'style', 'table'] # HTML tags we don't bother with

def html_to_text(path: Path) -> str:
    soup = BeautifulSoup(path.read_bytes(), 'lxml')

    for tag in soup.find_all(DROP_TAGS):
        tag.decompose() # Remove the tagged HTML including the text nested in

    return soup.get_text(separator='\n')

if __name__ == '__main__':
    from earnings_rag.config import settings
    text = html_to_text(settings.raw_dir / 'NVDA' / '2026-01-25.html')
    print(len(text))
    print(text[:3000])