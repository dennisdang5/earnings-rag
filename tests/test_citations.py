import json
from fakes import FakeClient, answer, tool_request
from earnings_rag.agent.citations import check_citations, revision_request
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
    assert check('Revenue was $130.5 billion [NVDA_revenue_FY2025].') == {'uncited': [], 'unknown_ids': [], 'unsupported': []}


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
    assert out == {'uncited': [], 'unknown_ids': [], 'unsupported': []}


def test_markdown_list_items_are_checked_separately():
    text = '1. Data Center grew 142% [NVDA_2025-01-26_0075].\n2. Automotive grew 55%.'
    assert check(text)['uncited'] == ['Automotive grew 55%.']


def test_an_id_no_tool_returned_is_flagged():
    out = check('Revenue was $130.5 billion [NVDA_revenue_FY2030].')
    assert out['unknown_ids'] == ['NVDA_revenue_FY2030']
    assert out['uncited'] == []


def test_the_refusal_sentence_is_fine():
    assert check('The provided filings do not address this.') == {'uncited': [], 'unknown_ids': [], 'unsupported': []}


def test_each_problem_gets_its_own_fix_in_the_revision_request():
    only_unknown = revision_request({'uncited': [], 'unknown_ids': ['COF_2023-12-31_0066']})
    assert 'Replace each with the id' in only_unknown and 'figure' not in only_unknown
    both = revision_request({'uncited': ['Revenue was $1 billion.'], 'unknown_ids': ['X_1']})
    assert 'Add the [id] that supports each figure' in both and 'Replace each with the id' in both


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


# --- figures a cited id does not state ---------------------------------------------------------------------------

VALUES = {'NVDA_revenue_FY2026_product_DataCenter': [193_737], 'NVDA_revenue_FY2026Q3_product_DataCenter': [51_215],
          'NVDA_revenue_FY2025': [130_497], 'NVDA_eps_diluted_FY2025': [2.94, 29.4]}
COMPUTED = {'calc_1': 62_314.0, 'calc_2': 0.462063}


def unsupported(text):
    return check_citations(text, set(VALUES) | set(COMPUTED), VALUES, COMPUTED)['unsupported']


def test_a_computed_figure_citing_the_fact_it_came_from_is_flagged():
    # the Q4 answer: FY minus three quarters, cited as the FY fact
    s = 'Q4 Data Center revenue was $62,314 million [NVDA_revenue_FY2026_product_DataCenter].'
    assert unsupported(s) == [('$62,314 million', s)]
    assert unsupported('Q4 Data Center revenue was $62,314 million [calc_1].') == []


def test_reported_figures_match_their_facts_at_the_precision_written():
    assert unsupported('Revenue was $130.5 billion [NVDA_revenue_FY2025].') == []
    assert unsupported('Revenue was $130,497 million [NVDA_revenue_FY2025].') == []
    assert unsupported('Diluted EPS was $2.94 [NVDA_eps_diluted_FY2025].') == []
    assert unsupported('Diluted EPS was $29.40 as reported [NVDA_eps_diluted_FY2025].') == []   # as_reported counts
    assert unsupported('Revenue was $131.5 billion [NVDA_revenue_FY2025].') == [('$131.5 billion',
                                                                                 'Revenue was $131.5 billion [NVDA_revenue_FY2025].')]


def test_a_percent_matches_a_calc_ratio_or_its_percent():
    assert unsupported('Gross margin was 46.21% [calc_2].') == []
    assert unsupported('Gross margin was 46.2% [calc_2].') == []
    assert len(unsupported('Gross margin was 46.21% [NVDA_revenue_FY2025].')) == 1   # a fact does not state a ratio


def test_sentences_citing_a_passage_or_nothing_are_not_checked():
    assert unsupported('Revenue rose 114% [NVDA_2025-01-26_0075].') == []   # prose figures: not matched
    assert unsupported('Revenue rose 114% [NVDA_2025-01-26_0075][NVDA_revenue_FY2025].') == []
    assert unsupported('Fiscal 2025 ended in January.') == []


def test_the_revision_names_the_unsupported_figure_and_says_to_cite_the_calc_id():
    text = revision_request({'uncited': [], 'unknown_ids': [],
                             'unsupported': [('$62,314 million', 'Q4 was $62,314 million [X].')]})
    assert '$62,314 million in: Q4 was $62,314 million [X].' in text and '[calc_1]' in text


def test_calculate_results_get_ids_the_answer_can_cite():
    from earnings_rag.agent.loop import record_numbers
    values, computed = {}, {}
    out = record_numbers('{"expression": "1 + 1", "result": 2.0}', values, computed)
    assert json.loads(out)['id'] == 'calc_1' and computed == {'calc_1': 2.0}
    assert json.loads(record_numbers('{"expression": "2 * 2", "result": 4.0}', values, computed))['id'] == 'calc_2'
    record_numbers('{"results": [{"id": "A", "value": 5, "as_reported": 50}]}', values, computed)
    assert values == {'A': [5, 50]}
    assert record_numbers('{"error": "bad"}', values, computed) == '{"error": "bad"}'
