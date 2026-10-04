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


# A figure with its parts: the number, its decimals (the precision it was written at), and a % or magnitude word
FIGURE_PARTS = re.compile(rf'(\$\s?)?(\d[\d,]*)(?:\.(\d+))?\s?(%|(?:million|billion|trillion)\b)?', re.IGNORECASE)
SCALE = {'million': 1, 'billion': 1e3, 'trillion': 1e6}  # dollar facts are shown to the model in USD millions


def _figures(sentence: str) -> list[tuple[str, float, float, bool]]:
    """(text, value, tolerance, is_percent) for each figure. Dollar amounts are in millions when a magnitude word is
    given and as written otherwise (per-share values, or millions copied as shown). The tolerance is half a unit of the
    last digit written, so "$130.5 billion" matches 130,497 and "46.2%" matches 46.21."""
    out = []
    for m in FIGURE_PARTS.finditer(sentence):
        dollar, whole, frac, unit = m.groups()
        if not dollar and not unit:
            continue  # a bare number (a year, a quarter, a count) is not a figure
        decimals = len(frac or '')
        number = float(whole.replace(',', '') + ('.' + frac if frac else ''))
        scale = 1.0 if not unit or unit == '%' else SCALE[unit.lower()]
        out.append((m.group(0).strip(), number * scale, 0.5 * 10 ** -decimals * scale + 1e-9, unit == '%'))
    return out


def unsupported_figures(sentence: str, ids: list[str], values: dict[str, list[float]],
                        computed: dict[str, float]) -> list[str]:
    """
    Figures in a sentence that no id cited in it produced: not a cited fact's value, not a cited calculation's result.
    Only sentences citing nothing but fact and calc ids are checked; a passage states figures in prose that we cannot
    match reliably, so a sentence citing one is left alone. Catches "$62,314 million [NVDA_revenue_FY2026_...]" when
    62,314 is FY minus three quarters, and "21.69% [the two revenue facts]". Magnitudes are compared, so a decrease
    written without its sign still matches.
    """
    if not ids or any(i not in values and i not in computed for i in ids):
        return []
    candidates = [abs(v) for i in ids for v in values.get(i, [])]
    results = [abs(computed[i]) for i in ids if i in computed]
    bad = []
    for text, number, tol, percent in _figures(sentence):
        pool = [r * k for r in results for k in (1, 100)] if percent else candidates + results
        if not any(abs(number - c) <= tol for c in pool):
            bad.append(text)
    return bad


def check_citations(answer: str, seen_ids: set[str], values: dict[str, list[float]] | None = None,
                    computed: dict[str, float] | None = None) -> dict:
    """
    Find what is wrong with an answer's citations without judging whether they support the claims:
    uncited: sentences stating a figure with no [id], unknown_ids: cited ids that no tool ever returned,
    unsupported: (figure, sentence) pairs whose figure none of the sentence's cited facts or calculations produced.
    seen_ids is every passage, fact and calc id returned so far; values maps fact ids to the numbers they state,
    computed maps calc ids to their results.
    """
    values, computed = values or {}, computed or {}
    uncited, unknown, unsupported = [], [], []
    for sentence in _sentences(answer):
        ids = _cited_ids(sentence)
        unknown.extend(i for i in ids if i not in seen_ids and i not in unknown)
        if not ids and FIGURE.search(sentence):
            uncited.append(sentence)
        unsupported.extend((f, sentence) for f in unsupported_figures(sentence, ids, values, computed))
    return {'uncited': uncited, 'unknown_ids': unknown, 'unsupported': unsupported}


# Ways of saying a filing is missing. Generous on purpose: a false "acknowledged" leaves the answer as it was before
# this check existed, a false "not acknowledged" costs a pointless revision.
UNAVAILABLE = re.compile(
    r"not available|unavailable|(?:does|do|did)\s?n[o']t exist|no (?:such )?(?:10-[QK]|filing|report|quarterly report|"
    r"annual report)|not (?:yet )?(?:been )?(?:filed|released|published|included)|not (?:in|part of) the|"
    r"(?:could|can)\s?(?:no|n')t (?:find|locate)|no (?:data|information|text|passages) (?:for|from|on)", re.IGNORECASE)


# Years as people write them: 2030 and FY2030, or two digits after FY, "fiscal (year)" or an apostrophe (FY30, FY'30,
# fiscal '30, Q2 '30). Two digits mean 20xx. Spelled-out years ("twenty thirty") and relative ones ("next year") are not
# read: the first is rare, the second needs today's date (see DECISIONS.md, missing-filing years).
FULL_YEAR = re.compile(r'(?<!\d)(20\d{2})(?!\d)')
SHORT_YEAR = re.compile(r"(?:\bFY\s?'?|\bfiscal(?:\s+year)?\s+'?|(?<![\w'])')(\d{2})\b", re.IGNORECASE)


def years_in(text: str) -> set[int]:
    """Every fiscal year the text mentions, written in full or as two digits."""
    return {int(y) for y in FULL_YEAR.findall(text)} | {2000 + int(y) for y in SHORT_YEAR.findall(text)}


def check_missing_filings(answer: str, question: str, missing: list[dict]) -> list[str]:
    """
    Labels of filings that search_filings reported missing, that the user asked for, and that the answer does not say
    are missing. "Asked for" means the filing's fiscal year is in the question (years_in): a period the model invented
    and then corrected on its own was never the user's request. "Says" means a sentence with an unavailability phrase
    that names the year, or follows a sentence that does. Both sides read years the same way, so a question about
    "FY30" is checked and an answer saying "fiscal '30 is not available" passes.
    """
    sentences = _sentences(answer)
    years = [years_in(s) for s in sentences]
    asked = years_in(question)
    out = []
    for m in missing:
        year = m.get('fiscal_year')
        if not year or year not in asked or m['label'] in out:
            continue
        said = any(UNAVAILABLE.search(s) and (year in years[i] or (i and year in years[i - 1]))
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
                     + '\nAdd the [id] that supports each figure: the calc id for a number from calculate, the fact '
                       'id for a number from get_financials (call a tool if you need the fact), or remove the figure.')
    if problems.get('unsupported'):
        parts.append('These figures are not stated by any id cited in their sentence:\n'
                     + '\n'.join(f'- {f} in: {s}' for f, s in problems['unsupported'])
                     + '\nA number computed with calculate cites that calculation\'s id (like [calc_1]), not the facts '
                       'it was computed from; a reported number cites the fact that states it.')
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
