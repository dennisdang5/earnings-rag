import re

# A figure is a number with a financial marker: $, %, or a magnitude word. Years, list markers and product numbers
# (RTX 40, H100) have none, so they are not flagged. Precision matters more than recall here: a false positive costs
# a pointless extra model call, a miss leaves us no worse than before the check existed.
NUMBER = r'\d[\d,]*(?:\.\d+)?'
FIGURE = re.compile(rf'\$\s?{NUMBER}|{NUMBER}\s?%|{NUMBER}\s*(?:million|billion|trillion)\b', re.IGNORECASE)
BRACKET = re.compile(r'\[([^\[\]]+)\]')
ID_LIKE = re.compile(r'^[A-Za-z0-9_.\-]+$')
LIST_MARKER = re.compile(r'^\s*(?:\d+[.)]|[-*•])\s+')


def _cited_ids(text: str) -> list[str]:
    """Every id inside square brackets, including comma-separated lists like [A_1, B_2]."""
    ids = []
    for group in BRACKET.findall(text):
        parts = [p.strip() for p in re.split(r'[,;]', group)]
        if all(ID_LIKE.match(p) for p in parts):
            ids.extend(parts)
    return ids


def _sentences(answer: str) -> list[str]:
    out = []
    for line in answer.splitlines():
        line = LIST_MARKER.sub('', line).strip()
        # split after . ! ? only when whitespace follows, so 130.5 stays whole
        out.extend(s.strip() for s in re.split(r'(?<=[.!?])\s+', line) if s.strip())
    return out


def check_citations(answer: str, seen_ids: set[str]) -> dict:
    """
    Find what is wrong with an answer's citations without judging whether they support the claims:
    uncited: sentences stating a figure with no [id], unknown_ids: cited ids that no tool ever returned.
    seen_ids is every passage and fact id returned so far.
    """
    uncited, unknown = [], []
    for sentence in _sentences(answer):
        ids = _cited_ids(sentence)
        unknown.extend(i for i in ids if i not in seen_ids and i not in unknown)
        if not ids and FIGURE.search(sentence):
            uncited.append(sentence)
    return {'uncited': uncited, 'unknown_ids': unknown}


def revision_request(problems: dict) -> str:
    parts = []
    if problems['uncited']:
        parts.append('These sentences state figures without a citation:\n'
                     + '\n'.join(f'- {s}' for s in problems['uncited']))
    if problems['unknown_ids']:
        parts.append('These cited ids were never returned by a tool: ' + ', '.join(problems['unknown_ids']) + '.')
    parts.append('Revise the answer: add the [id] that supports each figure (call a tool if you need the fact), '
                 'or remove the figure. Cite only ids that tools returned.')
    return '\n\n'.join(parts)
