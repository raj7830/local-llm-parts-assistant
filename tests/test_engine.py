"""Chat engine tests with a fake Ollama client (no model or server needed)."""

from types import SimpleNamespace as NS

from src.chatbot.engine import chat_stream


def chunk(content="", tool_calls=None, done=False):
    return NS(
        message=NS(content=content, thinking=None, tool_calls=tool_calls),
        done=done, eval_count=3 if done else None, eval_duration=1e9, prompt_eval_count=10,
    )


class FakeClient:
    """First call requests a tool, second call answers with text."""

    def __init__(self):
        self.calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            call = NS(function=NS(name="add", arguments={"a": 2, "b": 3}))
            return iter([chunk(tool_calls=[call], done=True)])
        assert kwargs["messages"][-1] == {"role": "tool", "tool_name": "add", "content": "5"}
        return iter([chunk("The answer "), chunk("is 5.", done=True)])


def test_tool_round_trip():
    messages = [{"role": "user", "content": "2+3?"}]
    events = list(chat_stream(FakeClient(), "m", messages, tools={"add": lambda a, b: str(a + b)}))
    kinds = [e.kind for e in events]
    assert kinds == ["stats", "tool", "content", "content", "stats"]
    assert "".join(e.text for e in events if e.kind == "content") == "The answer is 5."
    assert messages[-1] == {"role": "assistant", "content": "The answer is 5."}
    assert messages[1]["tool_calls"][0]["function"]["name"] == "add"


def test_tool_exception_is_reported_to_model():
    def boom():
        raise RuntimeError("bad")

    client = FakeClient()
    client.chat = lambda **kw: iter([chunk(tool_calls=[NS(function=NS(name="boom", arguments={}))], done=True)]) \
        if not any(m["role"] == "tool" for m in kw["messages"]) else iter([chunk("ok", done=True)])
    messages = [{"role": "user", "content": "x"}]
    events = list(chat_stream(client, "m", messages, tools={"boom": boom}))
    tool_event = next(e for e in events if e.kind == "tool")
    assert "RuntimeError: bad" in tool_event.data["result"]
