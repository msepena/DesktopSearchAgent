# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repo. The user-facing intro lives in [`README.md`](README.md) — keep them in sync when something visible there changes (stack, defaults, commands, milestones).

## Project

`DesktopSearchAgent` — a small local RAG utility. The user asks a natural-language question; the agent retrieves from a local vector index built over their configured folders and answers with Claude, citing sources. If the local index can't answer confidently, the pipeline falls back to web search (phase 2).

Bundled as a Python package `desktop_search` with a `dsa` CLI and a Streamlit web UI.

## Stack

- Python **3.11+**, dependencies managed by **`uv`** (lockfile committed).
- LLM: Anthropic **Claude Sonnet 4.6** (`claude-sonnet-4-6`) via the `anthropic` SDK, with prompt caching on the system block.
- Embeddings: **`fastembed`** running the HuggingFace `sentence-transformers/all-MiniLM-L6-v2` model via ONNX (CPU, ~80MB). We do **not** use `sentence-transformers`/`torch` — see "Platform note" below.
- Vector store: **ChromaDB** persisted to `./data/chroma/` (gitignored).
- File parsers: `pypdf`, `python-docx`, `python-pptx`, `openpyxl`. Plain-text/code is a flat `read_text`.
- CLI: `typer`. UI: `streamlit`. Config: a `Settings` Pydantic model + `config.yaml` (CLI flags override).
- Tests: `pytest`. PDF fixtures are generated in-process via `fpdf2` (dev-only dep).

## Architecture

```
src/desktop_search/
├── __init__.py        # __version__
├── config.py          # Settings (Pydantic) + load_settings(path)
├── loaders.py         # per-filetype text extraction + load_file dispatcher
├── indexer.py         # M3: walk → chunk → embed → upsert to Chroma
├── retriever.py       # M4: query → top-k Hit list
├── llm.py             # M5: Claude wrapper with prompt caching
├── pipeline.py        # M6: confidence gate; returns Response w/ source = local|web|none
├── cli.py             # typer app: index, ask, ui, version
└── app.py             # Streamlit UI (M8)

tests/
├── test_smoke.py      # CLI --help, version, package version
└── test_loaders.py    # per-loader + dispatcher tests with in-process fixtures
```

**Document shape**: `loaders.Document(text, source: Path, section: str | None, mtime: float)`. `section` is `"p.N"` for PDF, `"slide N"` for PPTX, sheet name for XLSX, `None` for DOCX/plain.

**Hit / Response shapes** are stubbed in `retriever.py` and `pipeline.py` — flesh them out in M4 and M6 respectively.

## Platform note (important when adding deps)

The dev machine is **Intel Mac, macOS 26, x86_64** (Core i7-9750H). In 2026, `torch>=2.3` and `onnxruntime>=1.21` no longer ship Intel-macOS wheels.

- Do **not** add `sentence-transformers`, `torch`, or anything that pulls them. Use `fastembed` for embeddings.
- The `onnxruntime>=1.18,<1.21` pin in `pyproject.toml` is load-bearing. Don't widen it without verifying wheels still resolve on Intel Mac.
- If a future dep transitively pulls `onnxruntime>=1.21` or `torch>=2.3`, the install will fail with "no source distribution or wheel for the current platform". Surface that to the user before changing course — they explicitly want local, open-source embeddings.

## Build & test

```bash
# install deps (creates .venv, reads uv.lock)
uv sync

# run the CLI (auto-uses .venv)
uv run dsa --help
uv run dsa version
uv run dsa index               # M3 — currently stubbed
uv run dsa ask "..."           # M6/M7 — currently stubbed
uv run dsa ui                  # M8 — currently stubbed

# tests
uv run pytest -q               # all
uv run pytest tests/test_loaders.py::test_load_pdf_one_doc_per_page
```

Tests are **pure pytest** (no Swift Testing here; that convention is from `FigmaDemo`). Generate fixtures in-process; do not check in binary blobs.

## Milestone discipline

Work is tracked as GitHub Issues #1–#10 (one per milestone). Each commit closing a milestone uses `Closes #N` in the body. The mirror checklist in `README.md` should be flipped to `[x]` in the same commit that closes the issue.

Acceptance criteria are written into each issue body — match them before declaring a milestone done.

## Conventions

- **Type hints** everywhere; prefer `NamedTuple` for record-shaped data and `pydantic` for validated config.
- **No torch / no sentence-transformers** (see platform note).
- **Errors at boundaries**: loaders raise on real parse failures; the indexer will catch and log per-file in M3. Don't wrap raises with broad `except` blocks inside loaders.
- **Tests run offline** — no network in unit tests. The `llm` module (M5) will need a mocked Anthropic client for tests.
- **Logging > prints** in library code. The CLI is the one place `typer.echo` is appropriate.
