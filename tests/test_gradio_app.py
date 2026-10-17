"""Gradio UI handler test with the fake Ollama client from test_engine (no server needed)."""

import gradio_app
from tests.test_engine import FakeClient


def test_respond_streams_tool_and_answer(monkeypatch):
    monkeypatch.setattr(gradio_app, "client", FakeClient())
    monkeypatch.setattr(gradio_app, "TOOLS", {"add": lambda a, b: str(a + b)})
    *_, (display, history, stats, box) = gradio_app.respond(
        "2+3?", [], [], "m", True, False, False, 0.3, "sys"
    )
    assert [m["role"] for m in display] == ["user", "assistant", "assistant"]
    assert display[1]["metadata"]["title"] == "🔧 Tool used: add"
    assert display[2]["content"] == "The answer is 5."
    assert history[0] == {"role": "system", "content": "sys"}
    assert stats.startswith("Last reply: 3 tokens") and box == ""


def test_empty_prompt_is_ignored():
    out = list(gradio_app.respond("  ", [], [], "m", True, False, False, 0.3, "sys"))
    assert out[0][0] == [] and out[0][1] == []
