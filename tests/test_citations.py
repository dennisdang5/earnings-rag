from fakes import FakeClient, answer, tool_request
from earnings_rag.agent.citations import check_citations
from earnings_rag.agent.loop import run_agent
from earnings_rag.agent.tools import Tool
from pydantic import BaseModel

SEEN = {'NVDA_revenue_FY2025', 'NVDA_revenue_FY2024', 'NVDA_2025-01-26_0075'}


def check(text, seen=SEEN):
    return check_citations(text, seen)


# --- the pure function -------------------------------------------------------

def test_figures_with_a_marker_are_flagged_when_uncited():
    for text in ('Revenue was $130.5 billion.', 'Revenue grew 114%.', 'It was 60,922 million.', 'EPS was $0.174.'):
        assert check(text)['uncited'] == [text]


def test_a_cited_figure_passes():
    assert check('Revenue was $130.5 billion [NVDA_revenue_FY2025].') == {'uncited': [], 'unknown_ids': []}


def test_years_list_markers_ids_and_product_numbers_are_not_figures():
    text = ('1. **Gaming**: Sales of the GeForce RTX 40 Series rose in fiscal 2025 and FY2024.\n'
            '- Hopper H100 shipped to customers.')
    assert check(text)['uncited'] == []


def test_digits_inside_an_id_are_not_figures():
    assert check('See [NVDA_2025-01-26_0075] for details.')['uncited'] == []


def test_each_sentence_is_judged_on_its_own():
    out = check('Revenue was $130.5 billion [NVDA_revenue_FY2025]. Growth was 114%.')
    assert out['uncited'] == ['Growth was 114%.']


def test_decimal_points_do_not_split_a_sentence():
    assert check('Revenue was $130.5 billion [NVDA_revenue_FY2025].')['uncited'] == []


def test_a_comma_separated_bracket_counts_as_citations():
    out = check('Revenue rose 114% [NVDA_revenue_FY2024, NVDA_revenue_FY2025].')
    assert out == {'uncited': [], 'unknown_ids': []}


def test_markdown_list_items_are_checked_separately():
    text = '1. Data Center grew 142% [NVDA_2025-01-26_0075].\n2. Automotive grew 55%.'
    assert check(text)['uncited'] == ['Automotive grew 55%.']


def test_an_id_no_tool_returned_is_flagged():
    out = check('Revenue was $130.5 billion [NVDA_revenue_FY2030].')
    assert out['unknown_ids'] == ['NVDA_revenue_FY2030']
    assert out['uncited'] == []


def test_the_refusal_sentence_is_fine():
    assert check('The provided filings do not address this.') == {'uncited': [], 'unknown_ids': []}


# --- in the loop -------------------------------------------------------------

class Args(BaseModel):
    text: str

FACTS = Tool('get', 'Fake facts.', Args, lambda a: {'results': [{'id': 'NVDA_revenue_FY2025'}]})


def test_a_cited_answer_costs_no_extra_model_call():
    client = FakeClient(tool_request('get', '{"text": "x"}'), answer('Revenue was $130.5 billion [NVDA_revenue_FY2025].'))
    result = run_agent('q', client, tools=[FACTS])
    assert len(client.requests) == 2
    assert not result.revised and result.uncited == []


def test_an_uncited_answer_is_sent_back_once_and_the_revision_is_returned():
    client = FakeClient(
        tool_request('get', '{"text": "x"}'),
        answer('Revenue was $130.5 billion.'),
        answer('Revenue was $130.5 billion [NVDA_revenue_FY2025].'),
    )
    result = run_agent('q', client, tools=[FACTS])
    assert result.revised and result.uncited == []
    assert result.answer.endswith('[NVDA_revenue_FY2025].')
    sent_back = client.requests[2]['messages']
    assert sent_back[-2] == {'role': 'assistant', 'content': 'Revenue was $130.5 billion.'}
    assert 'Revenue was $130.5 billion.' in sent_back[-1]['content'] and sent_back[-1]['role'] == 'user'


def test_the_revision_may_call_tools():
    client = FakeClient(
        tool_request('get', '{"text": "x"}'),
        answer('Revenue was $130.5 billion.'),
        tool_request('get', '{"text": "y"}', 'call_2'),
        answer('Revenue was $130.5 billion [NVDA_revenue_FY2025].'),
    )
    result = run_agent('q', client, tools=[FACTS])
    assert result.revised and len(result.trace) == 2 and result.uncited == []


def test_a_second_uncited_answer_is_accepted_and_flagged_not_looped():
    client = FakeClient(answer('Revenue was $130.5 billion.'), answer('Revenue was $130.5 billion.'))
    result = run_agent('q', client, tools=[FACTS])
    assert len(client.requests) == 2
    assert result.revised and result.uncited == ['Revenue was $130.5 billion.']


def test_an_invented_id_is_sent_back():
    client = FakeClient(answer('Revenue was $1 billion [NVDA_revenue_FY2030].'), answer('Revenue is not available.'))
    result = run_agent('q', client, tools=[FACTS])
    assert result.revised and result.unknown_ids == []


def test_the_forced_answer_is_checked_but_not_revised():
    client = FakeClient(tool_request('get', '{"text": "x"}'), answer('Revenue was $130.5 billion.'))
    result = run_agent('q', client, tools=[FACTS], max_steps=1)
    assert result.truncated and not result.revised
    assert result.uncited == ['Revenue was $130.5 billion.']
