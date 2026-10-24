"""Minimal local vector store backed by DuckDB (cosine similarity over embeddings)."""

import os
from collections.abc import Sequence
from functools import cache

import duckdb
from sentence_transformers import SentenceTransformer

STORE_PATH = os.getenv("RAG_STORE_PATH", "rag/vector_store.duckdb")
COLLECTION = os.getenv("RAG_COLLECTION", "legal_docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")


@cache
def get_embedder(model: str = EMBED_MODEL) -> SentenceTransformer:
    return SentenceTransformer(model)


def embed(texts: Sequence[str]) -> list[list[float]]:
    return get_embedder().encode(list(texts), show_progress_bar=False).tolist()


def _connect(path: str) -> duckdb.DuckDBPyConnection:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    con = duckdb.connect(path)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            collection VARCHAR,
            id VARCHAR,
            document VARCHAR,
            source VARCHAR,
            embedding FLOAT[],
            PRIMARY KEY (collection, id)
        )
        """
    )
    return con


def add(
    ids: Sequence[str],
    documents: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    sources: Sequence[str],
    *,
    collection: str = COLLECTION,
    path: str = STORE_PATH,
) -> None:
    """Insert (or replace) chunks with precomputed embeddings."""
    rows = [
        (collection, i, d, s, list(e))
        for i, d, e, s in zip(ids, documents, embeddings, sources, strict=True)
    ]
    with _connect(path) as con:
        con.executemany("INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?)", rows)


def query(
    query_embedding: Sequence[float],
    top_k: int = 3,
    *,
    collection: str = COLLECTION,
    path: str = STORE_PATH,
) -> list[tuple[str, str]]:
    """Return the top_k most similar (document, source) pairs."""
    if not os.path.exists(path):
        return []
    with _connect(path) as con:
        return con.execute(
            """
            SELECT document, source
            FROM chunks
            WHERE collection = ?
            ORDER BY list_cosine_similarity(embedding, ?::FLOAT[]) DESC
            LIMIT ?
            """,
            [collection, list(query_embedding), top_k],
        ).fetchall()
