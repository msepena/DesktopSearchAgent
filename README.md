# DesktopSearchAgent

A small local RAG (Retrieval-Augmented Generation) utility for your laptop. Ask natural-language questions; the agent searches your indexed files, returns a cited answer from local content, and falls back to web search when the local index can't answer confidently.

## How it works

1. **Indexer** — walks configured folders, extracts text from PDFs, Office docs, Markdown, plain text, and code, chunks it, and stores embeddings in a local vector store.
2. **Query pipeline** — embeds the question, retrieves top-k similar chunks from the vector store.
3. **Confidence gate** — if the best matches clear a similarity threshold, the LLM answers using those chunks; otherwise the query is forwarded to web search (phase 2).
4. **Answer generation** — Claude composes a cited answer from the retrieved context.
5. **UI** — CLI for power use, Streamlit web UI for casual querying.

## Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ (managed by `uv`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` via `fastembed` / ONNX (local, CPU) |
| Vector store | ChromaDB (file-backed at `./data/chroma/`) |
| LLM | Anthropic Claude Sonnet 4.6 (with prompt caching) |
| File parsers | `pypdf`, `python-docx`, `python-pptx`, `openpyxl` |
| CLI | `typer` |
| Web UI | `streamlit` |
| Config | `pydantic-settings` + `config.yaml` (CLI flags override) |
| Web search (phase 2) | TBD (Tavily / Brave / DuckDuckGo) |

## Defaults (tunable in `config.yaml`)

- Chunk size: **800 tokens**, **100-token overlap**
- Top-k: **6**
- Confidence threshold: cosine similarity ≥ **0.45** (below this triggers the web fallback placeholder)

## Project layout (planned)

```
DesktopSearchAgent/
├── README.md
├── pyproject.toml
├── config.yaml              # indexed folders, model names, thresholds
├── .env.example             # ANTHROPIC_API_KEY=
├── .gitignore
├── data/
│   └── chroma/              # local vector store (gitignored)
├── src/desktop_search/
│   ├── __init__.py
│   ├── config.py
│   ├── loaders.py           # per-filetype text extraction
│   ├── indexer.py           # walk → chunk → embed → store
│   ├── retriever.py         # query → top-k chunks + score
│   ├── llm.py               # Claude wrapper
│   ├── pipeline.py          # confidence gate, answer composition
│   ├── cli.py               # `dsa index` / `dsa ask` / `dsa ui`
│   └── app.py               # Streamlit UI
└── tests/
```

## CLI

```bash
dsa index                              # full index of folders in config.yaml
dsa index --path ~/Notes --path ~/Ref  # ad-hoc index of one or more paths
dsa index --watch                      # phase 2: live re-index on file changes
dsa ask "what was my Q3 OKR about latency?"
dsa ui                                 # launches Streamlit on localhost
dsa --help                             # list commands
```

All commands accept `--config PATH` (default `config.yaml`) to point at a
different config file. Set `ANTHROPIC_API_KEY` in `.env` before running `dsa ask`.

## Milestones / Todo

Tracked as GitHub Issues — checklist mirror here.

- [x] **M1** — Project scaffold (`pyproject.toml`, layout, `.env.example`, `config.yaml`, `.gitignore`)
- [x] **M2** — File loaders: PDF, docx, pptx, xlsx, md, txt, code
- [x] **M3** — Indexer: walk → chunk → embed → upsert to Chroma
- [x] **M4** — Retriever: top-k similarity search with scores
- [x] **M5** — Claude wrapper (`llm.py`) with prompt caching
- [x] **M6** — Pipeline: confidence gate + source-cited answer
- [x] **M7** — CLI commands: `index`, `ask`, `ui`
- [ ] **M8** — Streamlit chat UI
- [ ] **M9** — File-watcher mode (`index --watch`) — phase 2
- [ ] **M10** — Web search fallback (provider TBD) — phase 2

## Status

Current: **M7 complete** — `dsa index`, `dsa ask`, `dsa ui`, `dsa version` wired end-to-end. The CLI is the first usable surface: index a folder, ask a question, get a cited answer. 46/46 tests pass. Next up: **M8 — Streamlit UI**.

## Out of scope for v1

- File watcher (phase 2)
- Web search (phase 2)
- Per-document access controls
- OCR for image-only PDFs

## Platform note

The dev machine here is Intel Mac (macOS 26 + x86_64). In 2026, much of the ML
ecosystem (torch, recent `onnxruntime`) no longer ships Intel-macOS wheels. The
project sidesteps this by:

- Using **`fastembed`** instead of `sentence-transformers` — it runs the same
  HuggingFace `all-MiniLM-L6-v2` model through ONNX, no `torch` required.
- Pinning **`onnxruntime>=1.18,<1.21`** (versions before they dropped Intel-Mac
  wheels).

These pins are safe to keep on Apple Silicon and Linux too; they just unblock
Intel. Revisit if `fastembed` requires a newer `onnxruntime`.

## Setup (once implemented)

```bash
# clone and enter
git clone https://github.com/msepena/DesktopSearchAgent.git
cd DesktopSearchAgent

# install with uv
uv sync

# configure
cp .env.example .env          # add ANTHROPIC_API_KEY
$EDITOR config.yaml           # list folders to index

# build the index
uv run dsa index

# ask
uv run dsa ask "your question here"

# or launch the UI
uv run dsa ui
```
