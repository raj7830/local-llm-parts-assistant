"""UI-independent chat engine: streams replies from a local Ollama model and runs tool calls."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterator

import ollama

DEFAULT_MODEL = "gemma4:e2b-it-qat"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise assistant running fully offline on the user's computer. "
    "You have tools for a catalogue of fictitious electrical parts (IDs like A12): use them to look up "
    "alternative or matching parts instead of guessing, and present results as a short markdown table "
    "with ID, score and the key ratings. You also have tools to compute (working) hours between two "
    "timestamps. If you are unsure, say so."
)
MAX_TOOL_ROUNDS = 4


@dataclass
class Event:
    kind: str  # "thinking" | "content" | "tool" | "stats" | "error"
    text: str = ""
    data: dict = field(default_factory=dict)


def available_models(client: ollama.Client) -> list[str]:
    """Locally pulled chat models (embedding-only models are filtered out)."""
    names = [m.model for m in client.list().models]
    return sorted(n for n in names if "embed" not in n)


def chat_stream(
    client: ollama.Client,
    model: str,
    messages: list[dict],
    tools: dict[str, Callable[..., str]] | None = None,
    think: bool = False,
    options: dict | None = None,
) -> Iterator[Event]:
    """Stream a reply for ``messages``. Appends assistant/tool messages to ``messages`` in place.

    If the model requests tools, they are executed and the conversation continues
    automatically (up to MAX_TOOL_ROUNDS rounds).
    """
    tools = tools or {}
    for _ in range(MAX_TOOL_ROUNDS + 1):
        content, thinking, calls = "", "", []
        started = time.perf_counter()
        final = None
        for chunk in client.chat(
            model=model,
            messages=messages,
            tools=list(tools.values()) or None,
            think=think,
            options=options,
            stream=True,
        ):
            msg = chunk.message
            if msg.thinking:
                thinking += msg.thinking
                yield Event("thinking", msg.thinking)
            if msg.content:
                content += msg.content
                yield Event("content", msg.content)
            if msg.tool_calls:
                calls.extend(msg.tool_calls)
            if chunk.done:
                final = chunk

        assistant = {"role": "assistant", "content": content}
        if thinking:
            assistant["thinking"] = thinking
        if calls:
            assistant["tool_calls"] = [
                {"function": {"name": c.function.name, "arguments": dict(c.function.arguments)}} for c in calls
            ]
        messages.append(assistant)

        if final is not None and final.eval_count:
            secs = (final.eval_duration or 0) / 1e9
            yield Event(
                "stats",
                data={
                    "tokens": final.eval_count,
                    "tokens_per_s": final.eval_count / secs if secs else None,
                    "prompt_tokens": final.prompt_eval_count,
                    "wall_s": time.perf_counter() - started,
                },
            )

        if not calls:
            return
        for call in calls:
            name, args = call.function.name, dict(call.function.arguments)
            fn = tools.get(name)
            try:
                result = fn(**args) if fn else f'{{"error": "unknown tool {name}"}}'
            except Exception as exc:  # report tool failures back to the model
                result = f'{{"error": "{type(exc).__name__}: {exc}"}}'
            yield Event("tool", data={"name": name, "arguments": args, "result": result})
            messages.append({"role": "tool", "tool_name": name, "content": result})
    yield Event("error", "Stopped after too many consecutive tool calls.")
