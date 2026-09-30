import sys
from earnings_rag.pipeline import ask

def main() -> None:
    # Flag to allow you to show the sources that were used in generation of the answer
    show_sources = '--sources' in sys.argv
    args = [a for a in sys.argv[1:] if a != '--sources']

    if not args:
        print('usage: python ask.py "your question" [--sources]')
        sys.exit(1)

    question = ' '.join(args)
    result = ask(question)

    print(result['answer'])
    print('\nSources:')
    for i, hit in enumerate(result['sources'], start=1):
        print(f' [{i}] {hit["id"]} (distance {hit["distance"]:.3f})')
        if show_sources:
            print(f' {hit["text"][:2000]}\n')

if __name__ == '__main__':
    main()
