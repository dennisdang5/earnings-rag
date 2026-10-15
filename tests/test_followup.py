import json

from earnings_rag.agent.followup import format_history, rewrite
from earnings_rag.config import settings
from fakes import FakeClient, answer

HISTORY = [{'question': 'NVIDIA revenue fiscal 2025?', 'answer': 'It was $130.5 billion [NVDA_revenue_FY2025].'}]


def reply(question):
    return answer(json.dumps({'question': question}))


def test_no_history_means_no_model_call():
    client = FakeClient()
    out = rewrite('And Apple?', [], client)
    assert (out.question, out.changed) == ('And Apple?', False) and client.requests == []


def test_a_follow_up_is_replaced_by_a_standalone_question():
    client = FakeClient(reply("What was Apple's revenue in fiscal 2025?"))
    out = rewrite('And Apple?', HISTORY, client)
    assert out.changed and out.original == 'And Apple?' and out.question == "What was Apple's revenue in fiscal 2025?"
    sent = client.requests[0]
    assert sent['temperature'] == 0 and sent['response_format'] == {'type': 'json_object'}
    assert '[NVDA_revenue_FY2025]' in sent['messages'][1]['content'] and 'And Apple?' in sent['messages'][1]['content']


def test_a_question_that_already_stands_alone_is_unchanged():
    client = FakeClient(reply('what did apple report LAST YEAR?'))
    out = rewrite('What did Apple report last year?', HISTORY, client)
    assert not out.changed and out.question == 'What did Apple report last year?'


def test_failures_return_the_original_question():
    for bad in (answer('not json'), answer('{"other": 1}'), reply('   '), reply('x' * 1001), answer(None)):
        out = rewrite('And Apple?', HISTORY, FakeClient(bad))
        assert (out.question, out.changed) == ('And Apple?', False)

    class Broken:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise RuntimeError('api down')

    assert not rewrite('And Apple?', HISTORY, Broken()).changed


def test_history_is_trimmed_to_the_last_turns_and_answers_are_cut():
    turns = [{'question': f'q{i}', 'answer': 'a' * 5000} for i in range(6)]
    text = format_history(turns)
    assert 'q2' not in text and 'q3' in text and 'q5' in text
    assert text.count('a') == settings.followup_turns * settings.followup_answer_chars + text.count('Assistant')  # 'Assistant' has an 'a'
