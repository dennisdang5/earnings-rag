from pathlib import Path
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
import re
import tiktoken
import json
from earnings_rag.config import settings

warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning)

DROP_TAGS = ['script', 'style', 'table', 'ix:header', 'ix:hidden', 'ix:resources'] # HTML tags we don't bother with
INLINE_TAGS = ['span', 'b', 'i', 'em', 'strong', 'a', 'font', 'sup', 'sub']
NOISE_LINES = {'table of contents'}
_encoder = tiktoken.get_encoding('cl100k_base')

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

def is_noise(line: str) -> bool:
    if line.lower() in NOISE_LINES:
        return True
    if re.fullmatch(r'\d{1,3}', line):
        return True
    if re.fullmatch(r'[_\-—\s]+', line):
        return True
    return False

def normalize(text: str) -> str:
    lines = []
    for line in text.split('\n'):
        line = line.replace('\xa0', ' ') # Non-breaking space used by filers converted to real space
        line = re.sub(r'\s+', ' ', line).strip()
        if line and not is_noise(line):
            lines.append(line)

    return '\n'.join(lines)

def trim_front_matter(text: str, fallback_chars: int = 6000) -> str:
    m  = re.search(r'^Item\s*1\.?\s*Business', text, re.MULTILINE | re.IGNORECASE)
    if m:
        return text[m.start():]
    return text[fallback_chars:] # heading is not found then we cut a prefixed which is normally the table of contents


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError(f'overlap ({overlap}) must be less than size ({chunk_size})')

    tokens = _encoder.encode(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + chunk_size
        window = tokens[start:end]
        chunks.append(_encoder.decode(window))
        start = end - overlap

    return chunks

def chunk_document(path: Path, ticker: str) -> list[dict]:
    text = trim_front_matter(normalize(html_to_text(path)))
    pieces = chunk_text(text, settings.chunk_size_tokens, settings.chunk_overlap_tokens)

    records = []
    for i, piece in enumerate(pieces):
        records.append({
            'id': f'{ticker}_{path.stem}_{i:04d}',
            'ticker': ticker,
            'period': path.stem,
            'chunk_index': i,
            'text': piece
        })

    return records

def chunk_all() -> None:
    out_path = settings.chunks_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with out_path.open('w', encoding='utf-8') as f:
        for ticker in settings.tickers:
            for path in sorted((settings.raw_dir / ticker).glob('*.html')):
                records = chunk_document(path, ticker)
                for r in records:
                    f.write(json.dumps(r) + '\n')
                total += len(records)
                print(f'{ticker} {path.stem}: {len(records)} chunks')
    print(f'total: {total} -> {out_path}')

if __name__ == '__main__':
    from earnings_rag.config import settings
    chunk_all()