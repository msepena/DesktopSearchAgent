"""Indexer: walks folders, chunks loaded documents, embeds, persists in ChromaDB.

Idempotency: each chunk gets a deterministic id ``{path}#{chunk_index}`` and
each row's metadata carries the file's ``mtime``. A file with no mtime change
since the last run is skipped wholesale; a changed file has all its prior rows
deleted before fresh ones are upserted.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import chromadb
import tiktoken

from .config import Settings
from .loaders import Document, load_file


_TOKENIZER = tiktoken.get_encoding("cl100k_base")
_COLLECTION_NAME = "desktop_search"


@dataclass
class IndexStats:
    files_scanned: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    chunks_added: int = 0


@dataclass
class Chunk:
    text: str
    source: Path
    section: str | None
    mtime: float
    chunk_index: int


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedEmbedder:
    """Default embedder. Lazy-loads the fastembed model on first encode call."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self._model_name)
        return [list(v) for v in self._model.embed(texts)]


def _split_tokens(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")
    tokens = _TOKENIZER.encode(text)
    if not tokens:
        return []
    step = size - overlap
    out: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        out.append(_TOKENIZER.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += step
    return out


def chunk_documents(docs: list[Document], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    for doc in docs:
        for piece in _split_tokens(doc.text, chunk_size, overlap):
            chunks.append(
                Chunk(
                    text=piece,
                    source=doc.source,
                    section=doc.section,
                    mtime=doc.mtime,
                    chunk_index=idx,
                )
            )
            idx += 1
    return chunks


def _walk(folder: Path, ignore_patterns: list[str]) -> Iterator[Path]:
    if not folder.exists() or not folder.is_dir():
        return
    ignored = set(ignore_patterns)
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in ignored and not d.startswith(".")]
        for name in files:
            if name.startswith(".") or name in ignored:
                continue
            yield Path(root) / name


def get_collection(chroma_path: Path):
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _existing_mtimes(collection, source: str) -> set[float] | None:
    existing = collection.get(where={"source": source}, include=["metadatas"])
    metadatas = existing.get("metadatas") or []
    if not metadatas:
        return None
    return {m["mtime"] for m in metadatas if "mtime" in m}


def _index_file(path: Path, collection, embedder: Embedder, settings: Settings) -> int:
    docs = load_file(path)
    if not docs:
        return 0
    mtime = docs[0].mtime

    existing = _existing_mtimes(collection, str(path))
    if existing is not None:
        if existing == {mtime}:
            return 0
        collection.delete(where={"source": str(path)})

    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        return 0

    ids = [f"{path}#{c.chunk_index}" for c in chunks]
    texts = [c.text for c in chunks]
    embeddings = embedder.encode(texts)
    metadatas = [
        {
            "source": str(c.source),
            "section": c.section or "",
            "mtime": c.mtime,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]
    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(chunks)


def build_index(
    settings: Settings,
    paths: list[Path] | None = None,
    embedder: Embedder | None = None,
) -> IndexStats:
    folders = paths if paths else settings.folders
    stats = IndexStats()
    if not folders:
        return stats

    collection = get_collection(settings.chroma_path)
    if embedder is None:
        embedder = FastEmbedEmbedder(settings.embedding_model)

    for folder in folders:
        for file_path in _walk(Path(folder), settings.ignore_patterns):
            stats.files_scanned += 1
            added = _index_file(file_path, collection, embedder, settings)
            if added == 0:
                stats.files_skipped += 1
            else:
                stats.files_indexed += 1
                stats.chunks_added += added
    return stats
