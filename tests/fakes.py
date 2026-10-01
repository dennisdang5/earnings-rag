from types import SimpleNamespace


# --- scripted fake of the OpenAI client -------------------------------------

def answer(text, finish_reason='stop'):
    return SimpleNamespace(content=text, tool_calls=None, finish_reason=finish_reason)

def tool_request(name, arguments, call_id='call_1'):
    call = SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))
    return SimpleNamespace(content=None, tool_calls=[call], finish_reason='tool_calls')

class FakeClient:
    """Returns the scripted messages in order and records every request it receives."""
    def __init__(self, *messages):
        self.script = list(messages)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        msg = self.script.pop(0)
        # 100 input / 10 output tokens per call, so totals are easy to check
        return SimpleNamespace(
            choices=[SimpleNamespace(message=msg, finish_reason=msg.finish_reason)],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10),
        )
