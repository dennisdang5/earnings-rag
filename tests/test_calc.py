import pytest

from earnings_rag.calc import evaluate, MAX_EXPRESSION_CHARS


@pytest.mark.parametrize('expression, expected', [
    ('1 + 2', 3),
    ('2 + 3 * 4', 14),            # precedence
    ('(2 + 3) * 4', 20),          # parentheses
    ('-5 + 10', 5),               # unary minus
    ('+5', 5),
    ('7 / 2', 3.5),               # true division, not integer division
    ('2 ** 10', 1024),
    ('(60922 - 26974) / 26974 * 100', 125.8545265812),
    ('0.1 + 0.2', 0.3),           # rounding removes float noise
])
def test_evaluates_arithmetic(expression, expected):
    assert evaluate(expression) == expected


@pytest.mark.parametrize('expression', [
    "__import__('os').system('echo hi')",  # call
    '().__class__',                         # attribute access
    '[1, 2][0]',                            # list and subscript
    '"a" * 3',                              # string
    'True + 1',                             # bool is not a number
    '1j * 2',                               # complex literal
    'lambda: 1',
    '[x for x in (1, 2)]',
    'revenue * 2',                          # name
    '7 % 3',                                # operator outside the allowlist
    '7 // 2',
    '1 if 2 else 3',
])
def test_rejects_everything_outside_the_allowlist(expression):
    with pytest.raises(ValueError):
        evaluate(expression)


def test_thousands_separator_gets_a_hint():
    with pytest.raises(ValueError, match='thousands separators'):
        evaluate('26,974 + 1')


@pytest.mark.parametrize('expression', ['15%', '$100 + 5', 'not valid ((', ''])
def test_unparseable_input_gets_a_hint(expression):
    with pytest.raises(ValueError, match='Use only numbers'):
        evaluate(expression)


def test_division_by_zero():
    with pytest.raises(ValueError, match='Division by zero'):
        evaluate('1 / 0')


def test_overlong_expression_is_rejected():
    with pytest.raises(ValueError, match='longer than'):
        evaluate('1 + ' * MAX_EXPRESSION_CHARS + '1')


@pytest.mark.parametrize('expression', ['9 ** 9 ** 9 ** 9', '10 ** 400'])
def test_huge_results_fail_fast_instead_of_hanging(expression):
    with pytest.raises(ValueError, match='too large'):
        evaluate(expression)


def test_non_real_result_is_rejected():
    with pytest.raises(ValueError, match='not a real number'):
        evaluate('(-8) ** 0.5')


def test_literal_beyond_float_range_is_rejected():
    with pytest.raises(ValueError, match='not a finite number'):
        evaluate('1e400')
