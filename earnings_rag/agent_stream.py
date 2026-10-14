"""
Bridge from the synchronous agent loop to a stream of named events (the feed behind POST /agent/stream).

`run_agent` blocks until it has an answer, so it runs in a worker thread and reports progress through its `on_event`
callback into a queue; `stream_events` drains the queue as (event name, data) pairs. The answer is sent once, after
the loop's citation checks, never as a draft. Closing the generator (the client went away) stops the run at its next
event instead of letting it spend tokens for nobody.
"""
import json
import queue
import threading
from dataclasses import asdict
from typing import Callable, Iterator

from earnings_rag.agent.loop import run_agent

_DONE = object()


class Cancelled(Exception):
    """Raised inside the worker, from the event callback, to stop a run nobody is listening to."""


def summarize_result(result: str) -> dict:
    """What a live tool_result event carries: the ids the tool returned, how many, or its error. The whole result
    goes into `details` at the end; this keeps the live events small."""
    try:
        data = json.loads(result)
    except ValueError:
        return {'ids': [], 'count': 0, 'error': None}
    if not isinstance(data, dict):
        return {'ids': [], 'count': 0, 'error': None}
    rows = [r for r in data.get('results', []) if isinstance(r, dict)] if isinstance(data.get('results'), list) else []
    ids = [r['id'] for r in rows if 'id' in r] + ([data['id']] if isinstance(data.get('id'), str) else [])   # calc results
    return {'ids': ids, 'count': len(ids), 'error': data.get('error')}


def to_stream_event(event: dict) -> tuple[str, dict] | None:
    """Map a loop event to an SSE event name and payload; None for events the stream does not send live."""
    kind = event['type']
    if kind == 'model_call':
        return 'status', {'stage': 'writing' if event.get('forced') else 'thinking', 'step': event['step']}
    if kind == 'tool_call':
        return 'tool_call', {'step': event['step'], 'tool': event['tool'], 'arguments': event['arguments']}
    if kind == 'tool_result':
        return 'tool_result', {'step': event['step'], 'tool': event['tool'], 'new_results': event['new_results'],
                               **summarize_result(event['result'])}
    if kind == 'revising':
        return 'revising', {'problems': list(event['problems'])}
    return None


def stream_events(question: str, client, run_fn: Callable = run_agent) -> Iterator[tuple[str, dict]]:
    """
    Yield (event, data) for one agent run: status / tool_call / tool_result / revising while it works, then `answer`
    (the text), `details` (the whole result: trace, usage, flags, resolved periods) and `done`. A failure inside the
    run yields `error` instead, with a message safe to show; the exception is re-raised after the stream so the
    caller can log it.
    """
    events: queue.Queue = queue.Queue()
    stop = threading.Event()

    def on_event(event: dict) -> None:
        if stop.is_set():
            raise Cancelled()
        events.put(('event', event))

    def work() -> None:
        try:
            events.put(('result', run_fn(question, client, on_event=on_event)))
        except Cancelled:
            pass
        except Exception as e:   # reported to the stream, never lost in the thread
            events.put(('failed', e))
        finally:
            events.put((_DONE, None))

    threading.Thread(target=work, daemon=True).start()
    failure = None
    try:
        while True:
            kind, payload = events.get()
            if kind is _DONE:
                break
            if kind == 'event':
                mapped = to_stream_event(payload)
                if mapped:
                    yield mapped
            elif kind == 'result':
                yield 'answer', {'text': payload.answer}
                yield 'details', asdict(payload)
                yield 'done', {}
            else:
                failure = payload
                yield 'error', {'message': "Couldn't finish the answer. Try again in a moment."}
    finally:
        stop.set()   # a closed generator (client disconnect) ends the run at its next event
    if failure is not None:
        raise failure
