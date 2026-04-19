"""реализует retrieval для hybrid rag:
запускает dense-поиск, sparse-поиск или их объединение,
а в hybrid-режиме сливает результаты через rrf
и возвращает итоговый список релевантных чанков."""


from __future__ import annotations

from src.rag.naive_rag.retrieval import retrieve as dense_retrieve  # pyright: ignore[reportMissingImports]
from src.rag.hybrid_rag.sparse_index import search_sparse  # pyright: ignore[reportMissingImports]
from src.rag.hybrid_rag.fusion import fuse_rrf  # pyright: ignore[reportMissingImports]


def retrieve_dense(query: str, k: int) -> list[dict]:
    return dense_retrieve(query, k=k)


def retrieve_sparse(query: str, k: int) -> list[dict]:
    return search_sparse(query, k=k)


def retrieve(query: str, k: int, mode: str = "dense") -> list[dict]:
    if mode == "dense":
        return retrieve_dense(query, k=k)

    if mode == "sparse":
        return retrieve_sparse(query, k=k)

    if mode == "hybrid":
        dense_hits = retrieve_dense(query, k=k)
        sparse_hits = retrieve_sparse(query, k=k)
        return fuse_rrf(dense_hits, sparse_hits, k=k)

    raise ValueError(f"Unsupported hybrid retrieval mode: {mode}")