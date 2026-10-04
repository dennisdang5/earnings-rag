from earnings_rag.agent.tools import SearchFilingsArgs, search_filters


def filters(**kw):
    return search_filters(SearchFilingsArgs(query='q', **kw))


def test_default_searches_annual_reports_with_a_note():
    f = filters()
    assert (f['form'], f['fiscal_period']) == ('10-K', None)
    assert 'annual reports' in f['notes'][0]


def test_quarter_searches_that_10q():
    f = filters(fiscal_year=2027, period='Q2')
    assert (f['form'], f['fiscal_period'], f['notes']) == ('10-Q', 'Q2', [])


def test_q4_maps_to_the_10k_with_a_note():
    f = filters(fiscal_year=2026, period='Q4')
    assert (f['form'], f['fiscal_period']) == ('10-K', 'FY')
    assert 'Q4' in f['notes'][0]


def test_latest_searches_every_form():
    f = filters(latest=True)
    assert (f['form'], f['fiscal_period']) == (None, None)


def test_invalid_combinations_are_errors():
    assert 'error' in filters(latest=True, fiscal_year=2026)
    assert 'error' in filters(period='Q2')
