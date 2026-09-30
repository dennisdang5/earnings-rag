import sys

from earnings_rag.agent.loop import run_agent
from earnings_rag.llm import _llm


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python -m earnings_rag.agent "your question"')
        sys.exit(1)

    result = run_agent(' '.join(sys.argv[1:]), _llm())

    for t in result.trace:
        print(f'[step {t["step"]}] {t["tool"]}({t["arguments"]}) -> {len(t["result"])} chars')
    if result.truncated:
        print(f'[budget of {result.steps} steps spent: answer was forced]')
    print(f'\n{result.answer}')


if __name__ == '__main__':
    main()
