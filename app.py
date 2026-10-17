"""Local LLM chatbot (Task 1).  Start with:  streamlit run app.py"""

from __future__ import annotations

import json
from datetime import datetime

import ollama
import streamlit as st

from src.chatbot.engine import DEFAULT_MODEL, DEFAULT_SYSTEM_PROMPT, available_models, chat_stream
from src.chatbot.tools import TOOLS

st.set_page_config(page_title="Local Chatbot", page_icon="💬", layout="centered")

EXAMPLES = [
    "Suggest 5 alternatives for part A100",
    "I need a time lag 2A 250V 5x20mm glass part. What do you have?",
    "How many working hours are between 2022/02/18 16:00 and 2022/02/21 10:00?",
    "Explain the difference between fast blow and time lag fuses in two sentences.",
]


@st.cache_resource
def get_client() -> ollama.Client:
    return ollama.Client()


def init_state() -> None:
    st.session_state.setdefault("messages", [])  # full history sent to the model (incl. tool msgs)
    st.session_state.setdefault("pending", None)  # example prompt clicked by the user


def render_tool(event_data: dict) -> None:
    with st.expander(f"🔧 Tool used: `{event_data['name']}`", expanded=False):
        st.caption("Arguments")
        st.json(event_data["arguments"])
        st.caption("Result")
        try:
            st.json(json.loads(event_data["result"]))
        except json.JSONDecodeError:
            st.code(event_data["result"])


def render_history() -> None:
    pending_args: list[dict] = []  # arguments of the tool calls the last assistant message requested
    for msg in st.session_state.messages:
        role = msg["role"]
        if role == "system":
            continue
        if role == "assistant":
            pending_args = [c["function"]["arguments"] for c in msg.get("tool_calls", [])]
        if role == "tool":
            with st.chat_message("assistant", avatar="🔧"):
                render_tool({"name": msg.get("tool_name", "tool"),
                             "arguments": pending_args.pop(0) if pending_args else {},
                             "result": msg["content"]})
            continue
        if role == "assistant" and not msg.get("content"):
            continue  # pure tool-call request; the tool result is shown instead
        with st.chat_message(role):
            if msg.get("thinking") and st.session_state.get("show_thinking"):
                with st.expander("💭 Reasoning"):
                    st.markdown(msg["thinking"])
            st.markdown(msg["content"])


def export_markdown() -> str:
    lines = [f"# Chat export ({datetime.now():%Y-%m-%d %H:%M})", ""]
    for m in st.session_state.messages:
        if m["role"] in ("user", "assistant") and m.get("content"):
            lines += [f"**{m['role'].title()}:**", "", m["content"], ""]
    return "\n".join(lines)


def sidebar(client: ollama.Client) -> dict:
    with st.sidebar:
        st.header("⚙️ Settings")
        try:
            models = available_models(client)
        except Exception:
            st.error("Cannot reach Ollama. Start it with `ollama serve` (or open the Ollama app), "
                     f"then pull a model: `ollama pull {DEFAULT_MODEL}`.")
            st.stop()
        if not models:
            st.error(f"No models installed. Run `ollama pull {DEFAULT_MODEL}`.")
            st.stop()
        model = st.selectbox("Model", models, index=models.index(DEFAULT_MODEL) if DEFAULT_MODEL in models else 0)
        use_tools = st.toggle("Parts & time tools", value=True,
                              help="Let the model look up alternative parts and compute working hours.")
        think = st.toggle("Reasoning mode", value=False, help="Slower, but better for multi-step questions.")
        st.session_state.show_thinking = st.toggle("Show reasoning", value=False, disabled=not think)
        temperature = st.slider("Creativity (temperature)", 0.0, 1.5, 0.3, 0.1)
        with st.expander("System prompt"):
            system_prompt = st.text_area("System prompt", DEFAULT_SYSTEM_PROMPT, height=220,
                                         label_visibility="collapsed")

        col1, col2 = st.columns(2)
        if col1.button("🗑️ New chat", width="stretch"):
            st.session_state.messages = []
            st.rerun()
        col2.download_button("⬇️ Export", export_markdown(), file_name="chat.md", mime="text/markdown",
                             width="stretch", disabled=not st.session_state.messages)
        if "last_stats" in st.session_state:
            s = st.session_state.last_stats
            st.caption(f"Last reply: {s['tokens']} tokens"
                       + (f" · {s['tokens_per_s']:.1f} tok/s" if s.get("tokens_per_s") else "")
                       + f" · {s['wall_s']:.1f}s")
        st.caption("Runs 100% locally via Ollama — no data leaves this machine.")
    return {"model": model, "tools": TOOLS if use_tools else {}, "think": think,
            "temperature": temperature, "system_prompt": system_prompt}


def respond(client: ollama.Client, cfg: dict, prompt: str) -> None:
    messages = st.session_state.messages
    if not messages or messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": cfg["system_prompt"]})
    else:
        messages[0]["content"] = cfg["system_prompt"]
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.empty()
        status.caption("Thinking…")
        reasoning_box = st.expander("💭 Reasoning") if cfg["think"] and st.session_state.show_thinking else None
        reasoning_area = reasoning_box.empty() if reasoning_box else None
        answer_area = st.empty()
        answer, reasoning = "", ""
        try:
            for event in chat_stream(client, cfg["model"], messages, cfg["tools"], cfg["think"],
                                     {"temperature": cfg["temperature"]}):
                if event.kind == "thinking" and reasoning_area:
                    reasoning += event.text
                    reasoning_area.markdown(reasoning)
                elif event.kind == "content":
                    status.empty()
                    answer += event.text
                    answer_area.markdown(answer + "▌")
                elif event.kind == "tool":
                    status.caption(f"Used `{event.data['name']}`…")
                    render_tool(event.data)
                    answer = ""  # next round produces a fresh answer
                    answer_area = st.empty()
                elif event.kind == "stats":
                    st.session_state.last_stats = event.data
                elif event.kind == "error":
                    st.warning(event.text)
        except ollama.ResponseError as exc:
            st.error(f"Model error: {exc.error}")
        except Exception as exc:  # connection dropped etc.
            st.error(f"Could not get a reply: {exc}")
        status.empty()
        answer_area.markdown(answer)


def main() -> None:
    init_state()
    client = get_client()
    cfg = sidebar(client)

    st.title("💬 Local Chatbot")
    st.caption(f"Powered by **{cfg['model']}** on your own GPU via Ollama")

    if not st.session_state.messages:
        st.markdown("**Try one of these:**")
        cols = st.columns(2)
        for i, example in enumerate(EXAMPLES):
            if cols[i % 2].button(example, key=f"ex{i}", width="stretch"):
                st.session_state.pending = example
                st.rerun()

    render_history()
    prompt = st.chat_input("Ask anything…")
    if st.session_state.pending:
        prompt, st.session_state.pending = st.session_state.pending, None
    if prompt:
        respond(client, cfg, prompt)
        st.rerun()  # refresh sidebar stats / export button


if __name__ == "__main__":
    main()
