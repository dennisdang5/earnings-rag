import json
from datetime import date
from types import SimpleNamespace

from earnings_rag import chunking, pipeline
from earnings_rag.chunking import chunk_document, trim_front_matter
from earnings_rag.quarters import select_quarters, window_start
from earnings_rag.store import _where
from earnings_rag.xbrl import period_label


def filings(*report_dates):
    """get_filings() rows, newest first like the SEC feed."""
    return [{'report_date': d, 'accession': f'a-{d}', 'doc': 'x.htm', 'form': '10-Q'} for d in sorted(report_dates, reverse=True)]


# --- which 10-Qs --------------------------------------------------------------------------------------------------

def test_window_starts_one_year_before_the_oldest_10k_year_end():
    assert window_start('2023-01-29') == date(2022, 1, 29)
    assert window_start('2024-02-29') == date(2023, 2, 28)   # a leap day has no previous-year twin


def test_quarters_after_the_oldest_10ks_year_start_are_selected_oldest_first_including_those_after_the_newest_10k():
    nvda = filings('2021-10-31', '2022-05-01', '2022-07-31', '2022-10-30', '2023-04-30', '2026-04-26', '2026-07-26')
    chosen = [f['report_date'] for f in select_quarters(nvda, '2023-01-29')]
    # FY2023 (ended 2023-01-29) is the oldest 10-K: its quarters, every later one, and the quarters since the newest 10-K
    assert chosen == ['2022-05-01', '2022-07-31', '2022-10-30', '2023-04-30', '2026-04-26', '2026-07-26']


def test_a_filing_without_a_report_date_is_skipped():
    assert select_quarters([{'report_date': '', 'accession': 'x'}], '2023-01-29') == []


# --- cleaning ---------------------------------------------------------------------------------------------------------

TEN_Q = ('UNITED STATES SECURITIES AND EXCHANGE COMMISSION\nQuarterly report\nCover page text ' * 20
         + '\nPART I — FINANCIAL INFORMATION\nItem 1. Financial Statements\nNotes about the quarter.\n'
         + 'Item 2. Management Discussion\nRevenue grew.')


def test_a_10q_starts_at_the_financial_information_heading_not_at_a_fixed_cut():
    out = trim_front_matter(TEN_Q, form='10-Q')
    assert out.startswith('PART I — FINANCIAL INFORMATION') and 'Cover page text' not in out


def test_the_heading_variants_filers_use_are_all_found():
    for heading in ('Part I. Financial Information', 'PART I—FINANCIAL INFORMATION', 'PART I — FINANCIAL INFORMATION'):
        assert trim_front_matter(f'cover\n{heading}\nbody', form='10-Q').startswith(heading)


def test_a_10q_without_the_heading_falls_back_to_a_short_cut():
    text = 'x' * 5000
    assert len(trim_front_matter(text, form='10-Q')) == 3000


def test_10k_trimming_is_unchanged():
    assert trim_front_matter('cover\nItem 1. Business\nWe make chips.').startswith('Item 1. Business')
    assert trim_front_matter('x' * 7000) == 'x' * 1000


def test_chunk_records_carry_form_and_fiscal_label_and_keep_the_positional_id(tmp_path, monkeypatch):
    path = tmp_path / '2026-04-26.html'
    path.write_text('<html><body><p>PART I — FINANCIAL INFORMATION</p><p>' + 'Revenue grew a lot. ' * 200
                    + '</p></body></html>', encoding='utf-8')
    records = chunk_document(path, 'NVDA', '10-Q', (2027, 'Q1'))
    assert records and records[0]['id'] == 'NVDA_2026-04-26_0000'
    assert {(r['form'], r['fiscal_year'], r['fiscal_period']) for r in records} == {('10-Q', 2027, 'Q1')}
    tenk = tmp_path / '2026-01-25.html'
    tenk.write_text('<html><body><p>cover</p><p>Item 1. Business</p><p>' + 'We make chips. ' * 200 + '</p></body></html>',
                    encoding='utf-8')
    default = chunk_document(tenk, 'NVDA')
    assert default[0]['form'] == '10-K' and default[0]['fiscal_year'] is None and default[0]['id'] == 'NVDA_2026-01-25_0000'


# --- labels ------------------------------------------------------------------------------------------------------------

NVDA = [(2025, date(2024, 1, 29), date(2025, 1, 26)), (2026, date(2025, 1, 27), date(2026, 1, 25))]
AAPL = [(2025, date(2024, 9, 29), date(2025, 9, 27))]
COF = [(2025, date(2025, 1, 1), date(2025, 12, 31))]


def test_period_labels_for_a_january_a_september_and_a_december_year_end():
    assert period_label(date(2025, 7, 27), NVDA) == (2026, 'Q2')
    assert period_label(date(2025, 10, 26), NVDA) == (2026, 'Q3')
    assert period_label(date(2026, 1, 25), NVDA) == (2026, 'FY')      # the Q4 end is the fiscal year end
    assert period_label(date(2026, 4, 26), NVDA) == (2027, 'Q1')      # after the last 10-K: the year in progress
    assert period_label(date(2026, 6, 27), AAPL) == (2026, 'Q3')
    assert period_label(date(2026, 3, 31), COF) == (2026, 'Q1')
    assert period_label(date(2026, 6, 30), COF) == (2026, 'Q2')


def test_a_date_that_is_not_a_quarter_end_has_no_label():
    assert period_label(date(2025, 5, 15), NVDA) is None
    assert period_label(date(2025, 6, 1), NVDA) is None
    assert period_label(date(2025, 6, 1), []) is None


# --- filtered retrieval -----------------------------------------------------------------------------------------------

def test_a_filter_left_as_none_does_not_restrict():
    assert _where(None, None, None, None) == ('', [])
    assert _where('NVDA', None, None, None) == (' WHERE ticker = %s', ['NVDA'])
    assert _where('NVDA', '10-Q', 2027, 'Q1') == (
        ' WHERE ticker = %s AND form = %s AND fiscal_year = %s AND fiscal_period = %s', ['NVDA', '10-Q', 2027, 'Q1'])
    assert _where(None, '10-Q', None, None) == (' WHERE form = %s', ['10-Q'])


def test_retrieve_defaults_to_10ks_so_the_fixed_pipeline_ignores_the_new_quarters(monkeypatch):
    seen = {}
    monkeypatch.setattr(pipeline, 'search', lambda vector, **kw: seen.update(kw) or [])
    pipeline.retrieve('What risks does NVIDIA face?', query_vector=[0.0])
    assert seen['form'] == '10-K' and seen['ticker'] == 'NVDA'
    pipeline.retrieve('What risks does NVIDIA face?', query_vector=[0.0], form=None, fiscal_year=2027, fiscal_period='Q1')
    assert (seen['form'], seen['fiscal_year'], seen['fiscal_period']) == (None, 2027, 'Q1')


def test_index_run_can_embed_only_chunks_the_database_lacks(tmp_path, monkeypatch):
    path = tmp_path / 'chunks.jsonl'
    path.write_text('\n'.join(json.dumps({'id': i, 'text': f'text {i}'}) for i in ('a', 'b', 'c')), encoding='utf-8')
    monkeypatch.setattr(pipeline, 'settings', SimpleNamespace(chunks_path=path))
    monkeypatch.setattr(pipeline, 'chunk_ids', lambda: {'a', 'b'})
    embedded, stored = [], []
    monkeypatch.setattr(pipeline, 'embed_batched', lambda texts: embedded.extend(texts) or [[0.0]] * len(texts))
    monkeypatch.setattr(pipeline, 'upsert_chunks', lambda records, vectors: stored.extend(r['id'] for r in records))
    pipeline.build_index(new_only=True)
    assert embedded == ['text c'] and stored == ['c']
