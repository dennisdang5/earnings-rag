from pathlib import Path
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
import re

warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning)

DROP_TAGS = ['script', 'style', 'table', 'ix:header', 'ix:hidden', 'ix:resources'] # HTML tags we don't bother with
INLINE_TAGS = ['span', 'b', 'i', 'em', 'strong', 'a', 'font', 'sup', 'sub']

def html_to_text(path: Path) -> str:
    soup = BeautifulSoup(path.read_bytes(), 'lxml')

    for tag in soup.find_all(DROP_TAGS):
        tag.decompose() # Remove the tagged HTML including the text nested in

    # Anything the filer hid from browsers that is not content
    for tag in soup.select('[style*="display:none"], [style*="display: none"]'):
        tag.decompose()

    for tag in soup.find_all(INLINE_TAGS):
        tag.unwrap()

    for tag in soup.find_all(re.compile(r'^ix:')):
        tag.unwrap()

    soup.smooth() # Combine two adjacent strings into a single node

    return soup.get_text(separator='\n')

def normalize(text:str) -> str:
    lines = []
    for line in text.split('\n'):
        line = line.replace('\xa0', ' ') # Non-breaking space used by filers converted to real space
        line = re.sub(r'\s+', ' ', line).strip()
        if line:
            lines.append(line)

    return '\n'.join(lines)

if __name__ == '__main__':
    from earnings_rag.config import settings
    text = html_to_text(settings.raw_dir / 'NVDA' / '2026-01-25.html')
    text = normalize(text)
    print(len(text))
    print(text[8000:14000])