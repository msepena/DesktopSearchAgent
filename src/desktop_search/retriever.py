"""Retriever: embed a query, look up top-k chunks, expose normalized scores.

Chroma is configured with cosine space, so ``distance = 1 - cos_sim``. We
expose ``score = 1 - distance`` clamped to ``[0, 1]`` so downstream code (the
confidence gate in M6) sees a consistent unit-range similarity.
"""

from pathlib import Path
from typing import NamedTuple

from .config import Settings
from .indexer import Embedder, FastEmbedEmbedder, get_collection


class Hit(NamedTuple):
    text: str
    source: Path
    section: str | None
    score: float


def search(
    settings: Settings,
    query: str,
    k: int | None = None,
    embedder: Embedder | None = None,
) -> list[Hit]:
    k = k if k is not None else settings.top_k
    if k <= 0:
        return []

    collection = get_collection(settings.chroma_path)
    available = collection.count()
    if available == 0:
        return []

    if embedder is None:
        embedder = FastEmbedEmbedder(settings.embedding_model)
    query_embedding = embedder.encode([query])[0]

    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, available),
        include=["documents", "metadatas", "distances"],
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    hits: list[Hit] = []
    for text, meta, dist in zip(docs, metas, dists):
        score = max(0.0, min(1.0, 1.0 - float(dist)))
        section = meta.get("section") or None
        hits.append(
            Hit(
                text=text,
                source=Path(meta["source"]),
                section=section,
                score=score,
            )
        )
    return hits
