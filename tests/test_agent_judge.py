import json

import pytest

from earnings_rag.agent_judge import (evidence_items, groundedness_messages, judge_run, parse_groundedness,
                                      score_claim, split_claims)
from earnings_rag.judge_labels import build, select
from fakes import FakeClient, answer

SEARCH = json.dumps({'results': [{'id': 'NVDA_2025-01-26_0005', 'ticker': 'NVDA', 'form': '10-K', 'period': '2025-01-26',
                                  'fiscal_year': 2025, 'fiscal_period': 'FY', 'distance': 0.2, 'text': 'DLSS boosts frame rates.'}]})
FACT = json.dumps({'results': [{'id': 'AAPL_revenue_FY2026Q3', 'company': 'AAPL', 'metric': 'revenue', 'fiscal_year': 2026,
                                'period': 'Q3', 'value': 109417, 'unit': 'USD millions', 'derived': False}]})
CALC = json.dumps({'id': 'calc_1', 'expression': '1 + 1', 'result': 2})
TRACE = [{'tool': 'search_filings', 'result': SEARCH}, {'tool': 'get_financials', 'result': FACT},
         {'tool': 'calculate', 'result': CALC}, {'tool': 'search_filings', 'result': '{"error": "no"}'},
         {'tool': 'x', 'result': 'not json'}]

ANSWER = ('NVIDIA uses DLSS. It boosts frame rates.\n\nRay tracing helps too. Blackwell adds neural graphics '
          '[NVDA_2025-01-26_0005].')


def test_claims_are_numbered_sentences_and_keep_paragraph_citations():
    claims = split_claims(ANSWER)
    assert [c['n'] for c in claims] == [1, 2, 3, 4]
    assert claims[0]['paragraph_ids'] == [] and claims[3]['paragraph_ids'] == ['NVDA_2025-01-26_0005']
    assert claims[2]['ids'] == [] and claims[3]['ids'] == ['NVDA_2025-01-26_0005']   # cited in the paragraph, not in this sentence


def test_evidence_covers_passages_facts_calculations_and_errors_but_not_unparseable_results():
    items = evidence_items(TRACE)
    assert set(items) == {'NVDA_2025-01-26_0005', 'AAPL_revenue_FY2026Q3', 'calc_1', 'search_filings_error_1'}
    assert 'DLSS boosts frame rates.' in items['NVDA_2025-01-26_0005']
    assert '109417' in items['AAPL_revenue_FY2026Q3'] and '1 + 1 = 2' in items['calc_1']


def test_the_groundedness_prompt_holds_the_claims_and_every_evidence_item():
    claims = split_claims(ANSWER)
    user = groundedness_messages('q?', claims, evidence_items(TRACE))[1]['content']
    assert '1. NVIDIA uses DLSS.' in user and '[calc_1]' in user and '[AAPL_revenue_FY2026Q3]' in user


def entry(n, verdict='supported', ids=(), quote=''):
    return {'n': n, 'verdict': verdict, 'evidence_ids': list(ids), 'quote': quote}


def test_a_reply_must_cover_every_claim_in_order_with_known_verdicts():
    claims = split_claims(ANSWER)
    good = {'claims': [entry(i) for i in range(1, 5)]}
    assert len(parse_groundedness(good, claims)) == 4
    for bad in ({'claims': [entry(i) for i in range(1, 4)]}, {'claims': list(reversed(good['claims']))},
                {'claims': [entry(i, 'maybe') for i in range(1, 5)]}, {}):
        with pytest.raises(ValueError):
            parse_groundedness(bad, claims)


def test_citation_correct_compares_the_judges_ids_with_the_paragraph_citations():
    evidence = {'A': 'x', 'B': 'y'}
    claim = {'n': 1, 'text': 't', 'ids': [], 'paragraph_ids': ['A']}
    assert score_claim(claim, entry(1, 'supported', ['A']), evidence)['citation_correct'] is True
    assert score_claim(claim, entry(1, 'supported', ['B']), evidence)['citation_correct'] is False
    assert score_claim(claim, entry(1, 'unsupported'), evidence)['citation_correct'] is False
    assert score_claim({**claim, 'paragraph_ids': []}, entry(1, 'supported', ['A']), evidence)['citation_correct'] is None
    assert score_claim(claim, entry(1, 'not_a_claim'), evidence)['citation_correct'] is None


def test_ids_the_judge_invents_do_not_count_as_support():
    claim = {'n': 1, 'text': 't', 'ids': [], 'paragraph_ids': ['A']}
    scored = score_claim(claim, entry(1, 'supported', ['A']), evidence={'B': 'y'})
    assert scored['evidence_ids'] == [] and scored['citation_correct'] is False and scored['grounded']


def test_judge_run_makes_two_calls_and_scores_each_claim():
    record = {'id': 'q', 'run': 2, 'question': 'How does NVIDIA use AI for frame rates?', 'answer': ANSWER, 'trace': TRACE}
    client = FakeClient(
        answer(json.dumps({'claims': [entry(1, 'supported', ['NVDA_2025-01-26_0005'], 'DLSS boosts frame rates.'),
                                      entry(2), entry(3, 'unsupported'), entry(4, 'partial', ['NVDA_2025-01-26_0005'])]})),
        answer(json.dumps({'verdict': 'partial', 'off_topic_claims': [3]})))
    result = judge_run(record, client, model='judge-x')
    assert len(client.requests) == 2 and all(r['model'] == 'judge-x' and r['temperature'] == 0 for r in client.requests)
    assert [c['verdict'] for c in result['claims']] == ['supported', 'supported', 'unsupported', 'partial']
    assert result['claims'][3]['citation_correct'] is True and result['claims'][0]['citation_correct'] is None
    assert result['relevance'] == 'partial' and result['off_topic_claims'] == [3]
    assert (result['id'], result['run'], result['prompt_version']) == ('q', 2, 'v1')


# --- the blind label sheet ---------------------------------------------------

def run(qid, text, n=1, **over):
    return {'id': qid, 'run': n, 'question': f'{qid}?', 'answer': text, 'trace': TRACE, 'uncited': [], 'unknown_ids': [],
            'unsupported': [], 'unacknowledged_missing': [], 'slice_mismatch': [], **over}


def records():
    out = [run(f'q{i}', f'Question {i} has a long enough first sentence here. And a second sentence follows it.', n)
           for i in range(12) for n in (1, 2)]
    out += [run('aapl-iphone-q3-yoy', "Apple's iPhone revenue was $109,417 million [AAPL_revenue_FY2026Q3]."),
            run('aapl-cof-competition', 'Apple lists intellectual property as a competition risk. Capital One lists brand.'),
            run('nvda-last-quarter-dc', 'Data Center revenue grew a great deal last quarter.', unsupported=[['5%', 's']]),
            run('tesla-risks', 'The provided filings do not address this.')]
    return out


def test_selection_is_seeded_and_covers_each_stratum():
    a, b = select(records(), total=12), select(records(), total=12)
    assert [(s['record']['id'], s['claim']['n']) for s in a] == [(s['record']['id'], s['claim']['n']) for s in b]
    assert len(a) == 12 and len({s['claim']['text'] for s in a}) == 12
    assert {s['stratum'] for s in a} >= {'slice_error', 'drift', 'flagged', 'refusal', 'clean'}
    clean = [s['record']['id'] for s in a if s['stratum'] == 'clean']
    assert len(clean) == len(set(clean))      # one per question


def test_the_sheet_is_blind_and_unlabelled_and_the_metadata_has_the_strata():
    sheet, meta = build(select(records(), total=12))
    for e in sheet:
        assert set(e) == {'label_id', 'question', 'answer', 'claim', 'cited_in_paragraph', 'evidence', 'support',
                          'addresses_question'}
        assert e['support'] == '' and e['addresses_question'] == ''
    assert {m['label_id'] for m in meta} == {e['label_id'] for e in sheet}
    assert {m['split'] for m in meta} == {'dev', 'test'} and all('stratum' in m for m in meta)


def test_tool_errors_are_evidence_for_claims_about_what_is_available():
    trace = [{'tool': 'get_financials', 'result': '{"error": "COF does not report gross_profit. Available: revenue"}'},
             {'tool': 'search_filings', 'result': '{"error": "No FY2031 10-K. Filings with text: FY2025"}'}]
    items = evidence_items(trace)
    assert set(items) == {'get_financials_error_1', 'search_filings_error_2'}
    assert 'does not report gross_profit' in items['get_financials_error_1']


def test_uncited_passages_show_the_sentences_most_like_the_claim():
    from earnings_rag.judge_labels import excerpt
    text = ('Intro about nothing. We compete on the rates we pay on deposits and our customer service. '
            'Unrelated sentence about leases.')
    out = excerpt(text, 'The company competes on rates and customer service.')
    assert 'customer service' in out and 'leases' not in out
    assert excerpt('nothing matches here at all', 'zzzz qqqq') == 'nothing matches here at all'


# --- agreement and the human review set ---------------------------------------

def test_compare_counts_exact_binary_and_relevance_agreement():
    from earnings_rag.judge_agreement import compare
    labels = {i: {'support': s, 'addresses_question': a} for i, s, a in
              [('L1', 'supported', 'yes'), ('L2', 'partial', 'yes'), ('L3', 'unsupported', 'no'), ('L4', 'supported', 'yes')]}
    judged = {i: {'support': s, 'addresses_question': a} for i, s, a in
              [('L1', 'supported', 'yes'), ('L2', 'supported', 'yes'), ('L3', 'unsupported', 'yes'), ('L4', 'supported', 'yes')]}
    r = compare(labels, judged)
    assert (r['support_agree'], r['binary_agree'], r['addresses_agree']) == (0.75, 0.75, 0.75)
    assert r['matrix']['partial']['supported'] == 1 and r['disagreements'] == ['L2', 'L3']


def test_the_review_set_has_every_disagreement_and_a_seeded_sample_of_agreements():
    from earnings_rag.judge_agreement import review_set
    ids = [f'L{i:02d}' for i in range(1, 31)]
    a = review_set(['L03', 'L09'], ids, agreed_sample=5)
    assert a == review_set(['L03', 'L09'], ids, agreed_sample=5) and len(a) == 7 and {'L03', 'L09'} <= set(a)
    assert a == sorted(a)
