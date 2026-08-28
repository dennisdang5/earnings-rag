import sys
from earnings_rag.pipeline import ask

def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python ask.py "your question"')
        sys.exit(1)

    question = ' '.join(sys.argv[1:])
    result = ask(question)

    print(result['answer'])
    print('\nSources:')
    for i, hit in enumerate(result['sources'], start=1):
        print(f' [{i}] {hit['ticker']} {hit['period']} (distance {hit['distance']:.3f})')


    # Flag to allow you to show the sources that were used in generation of the answer
    show_sources = "--sources" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--sources"]
    question = " ".join(args)
    for i, hit in enumerate(result["sources"], start=1):
        print(f"[{i}] {hit['id']} (distance {hit['distance']:.3f})")
        if show_sources:
            print(f" {hit['text'][:2000]}\n")

if __name__ == '__main__':
    main()