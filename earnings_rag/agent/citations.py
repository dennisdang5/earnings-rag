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


# Ways of saying a filing is missing. Generous on purpose: a false "acknowledged" leaves the answer as it was before
# this check existed, a false "not acknowledged" costs a pointless revision.
UNAVAILABLE = re.compile(
    r"not available|unavailable|(?:does|do|did)\s?n[o']t exist|no (?:such )?(?:10-[QK]|filing|report|quarterly report|"
    r"annual report)|not (?:yet )?(?:been )?(?:filed|released|published|included)|not (?:in|part of) the|"
    r"(?:could|can)\s?(?:no|n')t (?:find|locate)|no (?:data|information|text|passages) (?:for|from|on)", re.IGNORECASE)


def check_missing_filings(answer: str, question: str, missing: list[dict]) -> list[str]:
    """
    Labels of filings that search_filings reported missing, that the user asked for, and that the answer does not say
    are missing. "Asked for" means the filing's fiscal year is written in the question: a period the model invented and
    then corrected on its own was never the user's request. "Says" means a sentence with an unavailability phrase that
    names the year, or follows a sentence that does.
    """
    sentences = _sentences(answer)
    out = []
    for m in missing:
        year = str(m.get('fiscal_year') or '')
        if not year or year not in question or m['label'] in out:
            continue
        said = any(UNAVAILABLE.search(s) and (year in s or (i and year in sentences[i - 1]))
                   for i, s in enumerate(sentences))
        if not said:
            out.append(m['label'])
    return out


def revision_request(problems: dict) -> str:
    # Each problem gets its own fix. A single figure-only instruction made the model answer an unknown passage id by
    # fetching unrelated financials and leaving the bad id in place (competition question, agent/10q-search-tool).
    parts = []
    if problems['uncited']:
        parts.append('These sentences state figures without a citation:\n'
                     + '\n'.join(f'- {s}' for s in problems['uncited'])
                     + '\nAdd the [id] that supports each figure (call a tool if you need the fact), or remove the figure.')
    if problems['unknown_ids']:
        parts.append('These cited ids were never returned by a tool: ' + ', '.join(problems['unknown_ids']) + '.'
                     '\nReplace each with the id of the tool result that states the claim (the ids are in the results '
                     'above, no new tool call is needed), or remove the claim.')
    if problems.get('unacknowledged_missing'):
        parts.append('You searched for ' + ', '.join(problems['unacknowledged_missing']) + ' and the tool reported '
                     'each as not available, but the answer does not say so.\nBegin the answer by saying the filing '
                     'asked for is not available. If you answer from another filing, name it; keep the rest of the '
                     'answer.')
    parts.append('Revise the answer. Cite only ids that tools returned.')
    return '\n\n'.join(parts)
