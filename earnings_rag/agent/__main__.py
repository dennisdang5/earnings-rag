import sys

from earnings_rag.agent.loop import run_agent
from earnings_rag.llm import _llm


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python -m earnings_rag.agent "your question"')
        sys.exit(1)

    result = run_agent(' '.join(sys.argv[1:]), _llm())

    for t in result.trace:
        new = '' if t['new_results'] is None else f', {t["new_results"]} new'
        print(f'[step {t["step"]}] {t["tool"]}({t["arguments"]}) -> {len(t["result"])} chars{new}')
    if result.truncated:
        print(f'[budget of {result.steps} steps spent: answer was forced]')
    print(f'\n{result.answer}')
    if result.cut_off:
        print('\n[answer cut off at max_tokens]')
    print(f'\n[{result.steps} model calls, {len(result.trace)} tool calls, '
          f'{result.input_tokens} in / {result.output_tokens} out tokens]')


if __name__ == '__main__':
    main()
