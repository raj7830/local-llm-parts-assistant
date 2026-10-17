"""Local LLM chatbot (Task 1).  Start with:  python app.py"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime

import gradio as gr
import ollama

from src.chatbot.engine import DEFAULT_MODEL, DEFAULT_SYSTEM_PROMPT, available_models, chat_stream
from src.chatbot.tools import TOOLS

EXAMPLES = [
    "Suggest 5 alternatives for part A100",
    "I need a time lag 2A 250V 5x20mm glass part. What do you have?",
    "How many working hours are between 2022/02/18 16:00 and 2022/02/21 10:00?",
    "Explain the difference between fast blow and time lag fuses in two sentences.",
]

client = ollama.Client()


def list_models() -> tuple[list[str], str | None, str]:
    """(models, default, status markdown) - never raises, so the UI can always start."""
    try:
        models = available_models(client)
    except Exception:
        return [], None, ("⚠️ **Cannot reach Ollama.** Start it with `ollama serve` (or open the Ollama app), "
                          f"then pull a model: `ollama pull {DEFAULT_MODEL}`, and reload this page.")
    if not models:
        return [], None, f"⚠️ **No models installed.** Run `ollama pull {DEFAULT_MODEL}` and reload this page."
    default = DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]
    return models, default, "Runs 100% locally via Ollama — no data leaves this machine."


def tool_message(data: dict) -> dict:
    try:
        result = json.dumps(json.loads(data["result"]), indent=2)
    except json.JSONDecodeError:
        result = data["result"]
    body = (f"**Arguments**\n```json\n{json.dumps(data['arguments'], indent=2)}\n```\n"
            f"**Result**\n```json\n{result}\n```")
    return {"role": "assistant", "content": body,
            "metadata": {"title": f"🔧 Tool used: {data['name']}", "status": "done"}}


def respond(prompt, display, history, model, use_tools, think, show_thinking, temperature, system_prompt):
    """Generator: streams (chatbot display, llm history, stats, cleared textbox)."""
    prompt = (prompt or "").strip()
    if not prompt:
        yield display, history, gr.skip(), ""
        return
    if not model:
        raise gr.Error(list_models()[2])

    history = list(history)
    if history and history[0]["role"] == "system":
        history[0] = {"role": "system", "content": system_prompt}
    else:
        history.insert(0, {"role": "system", "content": system_prompt})
    history.append({"role": "user", "content": prompt})
    display = display + [{"role": "user", "content": prompt}]
    yield display, history, "Thinking…", ""

    reasoning_idx = answer_idx = None
    stats = ""
    try:
        for event in chat_stream(client, model, history, TOOLS if use_tools else {}, think,
                                 {"temperature": temperature}):
            if event.kind == "thinking" and show_thinking:
                if reasoning_idx is None:
                    display.append({"role": "assistant", "content": "",
                                    "metadata": {"title": "💭 Reasoning", "status": "pending"}})
                    reasoning_idx = len(display) - 1
                display[reasoning_idx]["content"] += event.text
            elif event.kind == "content":
                if reasoning_idx is not None:
                    display[reasoning_idx]["metadata"]["status"] = "done"
                    reasoning_idx = None
                if answer_idx is None:
                    display.append({"role": "assistant", "content": ""})
                    answer_idx = len(display) - 1
                display[answer_idx]["content"] += event.text
            elif event.kind == "tool":
                display.append(tool_message(event.data))
                answer_idx = reasoning_idx = None  # the next round starts a fresh message
            elif event.kind == "stats":
                s = event.data
                stats = (f"Last reply: {s['tokens']} tokens"
                         + (f" · {s['tokens_per_s']:.1f} tok/s" if s.get("tokens_per_s") else "")
                         + f" · {s['wall_s']:.1f}s")
            elif event.kind == "error":
                gr.Warning(event.text)
            yield display, history, gr.skip(), ""
    except ollama.ResponseError as exc:
        raise gr.Error(f"Model error: {exc.error}")
    except Exception as exc:  # connection dropped etc.
        raise gr.Error(f"Could not get a reply: {exc}")
    yield display, history, stats, ""


def export_chat(history):
    lines = [f"# Chat export ({datetime.now():%Y-%m-%d %H:%M})", ""]
    for m in history:
        if m["role"] in ("user", "assistant") and m.get("content"):
            lines += [f"**{m['role'].title()}:**", "", m["content"], ""]
    f = tempfile.NamedTemporaryFile("w", suffix=".md", prefix="chat-", delete=False)
    f.write("\n".join(lines))
    f.close()
    return f.name


def build() -> gr.Blocks:
    models, default, status = list_models()
    with gr.Blocks(title="Local Chatbot", fill_height=True) as demo:
        history = gr.State([])  # full LLM history incl. system/tool messages

        with gr.Row(equal_height=False):
            with gr.Column(scale=4):
                gr.Markdown(f"# 💬 Local Chatbot\nPowered by local LLMs on your own GPU via Ollama")
                chatbot = gr.Chatbot(
                    height=560,
                    placeholder="**Ask anything** — or try an example below.",
                    buttons=["copy"],
                    avatar_images=(None, None),
                )
                with gr.Row():
                    textbox = gr.Textbox(placeholder="Ask anything…", show_label=False, scale=8,
                                         autofocus=True, submit_btn=True)
                gr.Examples(EXAMPLES, inputs=textbox, label="Try one of these")

            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### ⚙️ Settings")
                model = gr.Dropdown(models, value=default, label="Model", interactive=bool(models))
                use_tools = gr.Checkbox(True, label="Parts & time tools",
                                        info="Look up alternative parts and compute working hours.")
                think = gr.Checkbox(False, label="Reasoning mode", info="Slower, better for multi-step questions.")
                show_thinking = gr.Checkbox(False, label="Show reasoning")
                temperature = gr.Slider(0.0, 1.5, value=0.3, step=0.1, label="Creativity (temperature)")
                with gr.Accordion("System prompt", open=False):
                    system_prompt = gr.Textbox(DEFAULT_SYSTEM_PROMPT, lines=8, show_label=False)
                with gr.Row():
                    clear = gr.Button("🗑️ New chat", size="sm")
                    export = gr.DownloadButton("⬇️ Export", size="sm")
                stats = gr.Markdown("")
                gr.Markdown(status)

        inputs = [textbox, chatbot, history, model, use_tools, think, show_thinking, temperature, system_prompt]
        outputs = [chatbot, history, stats, textbox]
        textbox.submit(respond, inputs, outputs, concurrency_limit=1)
        clear.click(lambda: ([], [], ""), None, [chatbot, history, stats])
        export.click(export_chat, history, export)
    return demo


if __name__ == "__main__":
    build().launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())
