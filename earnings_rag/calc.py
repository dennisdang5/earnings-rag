import ast
import math
import operator

MAX_EXPRESSION_CHARS = 200

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}

SYNTAX_HELP = ('Use only numbers, + - * / ** and parentheses. No %, $, units or thousands separators: '
               'write percentages as decimals or multiply by 100 explicitly.')


def _eval(node: ast.AST) -> float:
    """
    Evaluate one AST node. Anything not explicitly allowed is rejected, so names, calls, attributes and
    subscripts (the ways to escape a sandbox) cannot appear.
    """
    if isinstance(node, ast.Constant):
        # type() rather than isinstance(): bool is a subclass of int and must not count as a number
        if type(node.value) not in (int, float):
            raise ValueError(f'Only numbers are allowed, not {type(node.value).__name__}. {SYNTAX_HELP}')
        return float(node.value)  # floats everywhere: 9**9**9**9 overflows quickly instead of building a huge int

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        value = _BINARY_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(value, complex):  # (-8) ** 0.5 returns complex in Python 3
            raise ValueError('Result is not a real number.')
        return value

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))

    if isinstance(node, ast.Tuple):
        raise ValueError('Got a comma-separated list. Remove thousands separators (write 26974, not 26,974).')

    raise ValueError(f'{type(node).__name__} is not allowed. {SYNTAX_HELP}')


def evaluate(expression: str) -> float:
    """
    Safely evaluate an arithmetic expression. Raises ValueError with a message meant for the model to read.
    """
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise ValueError(f'Expression longer than {MAX_EXPRESSION_CHARS} characters.')

    try:
        tree = ast.parse(expression.strip(), mode='eval')
    except SyntaxError:
        raise ValueError(f'Could not parse the expression. {SYNTAX_HELP}') from None

    try:
        result = _eval(tree.body)
    except ZeroDivisionError:
        raise ValueError('Division by zero.') from None
    except OverflowError:
        raise ValueError('Result is too large.') from None

    if not math.isfinite(result):
        raise ValueError('Result is not a finite number.')

    return round(result, 10)  # removes float noise like 0.30000000000000004; display rounding is left to the caller
