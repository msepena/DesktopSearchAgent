"""Streamlit chat UI. Reuses pipeline.ask — no duplicated retrieval/LLM logic."""

from datetime import datetime

import streamlit as st

from desktop_search.config import load_settings
from desktop_search.indexer import get_collection
from desktop_search.pipeline import ask


def _render_sidebar(settings) -> None:
    st.sidebar.header("Index")
    try:
        collection = get_collection(settings.chroma_path)
        st.sidebar.metric("Chunks indexed", collection.count())
        sample = collection.get(limit=10_000, include=["metadatas"])
        mtimes = [
            m.get("mtime")
            for m in (sample.get("metadatas") or [])
            if isinstance(m.get("mtime"), (int, float))
        ]
        if mtimes:
            last = datetime.fromtimestamp(max(mtimes))
            st.sidebar.caption(f"Last file mtime: {last:%Y-%m-%d %H:%M}")
        else:
            st.sidebar.caption("No files indexed yet — run `dsa index`.")
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.sidebar.error(f"Index unavailable: {exc}")

    st.sidebar.header("Model")
    st.sidebar.write(f"LLM: `{settings.llm_model}`")
    st.sidebar.write(f"Embeddings: `{settings.embedding_model}`")
    st.sidebar.write(f"Threshold: `{settings.confidence_threshold}`")
    st.sidebar.write(f"Top-k: `{settings.top_k}`")


def _format_sources(hits) -> list[str]:
    lines: list[str] = []
    for h in hits:
        line = f"- `{h.source}`"
        if h.section:
            line += f" *({h.section})*"
        line += f" — score `{h.score:.2f}`"
        lines.append(line)
    return lines


def _render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                st.markdown("\n".join(msg["sources"]))
        if msg.get("source_label") and msg["source_label"] != "local":
            st.caption(f"answer source: {msg['source_label']}")


def main() -> None:
    st.set_page_config(page_title="DesktopSearchAgent", layout="wide")
    st.title("DesktopSearchAgent")
    st.caption("Ask a question about your indexed files. Answers are cited from local content.")

    settings = load_settings()
    _render_sidebar(settings)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        _render_message(msg)

    prompt = st.chat_input("Ask a question about your indexed files...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    _render_message(st.session_state.messages[-1])

    with st.chat_message("assistant"):
        with st.spinner("Searching local files..."):
            response = ask(settings, prompt)
        st.markdown(response.answer)
        sources = _format_sources(response.hits)
        if sources:
            with st.expander(f"Sources ({len(sources)})"):
                st.markdown("\n".join(sources))
        if response.source != "local":
            st.caption(f"answer source: {response.source}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.answer,
            "sources": sources,
            "source_label": response.source,
        }
    )


if __name__ == "__main__":
    main()
