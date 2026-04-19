"""реализует sparse retrieval на основе bm25:
строит статистику по токенам чанков, вычисляет bm25-оценки для запроса,
ранжирует релевантные чанки
и возвращает top-k результатов для hybrid rag."""


from __future__ import annotations

import math
import re

from collections import Counter
from functools import lru_cache

from src.rag.common.text_utils import to_pos_int  # pyright: ignore[reportMissingImports]
from src.rag.hybrid_rag.sparse_index import load_sparse_chunks  # pyright: ignore[reportMissingImports]

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")


def _tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    return [t for t in _TOKEN_RE.findall(text) if len(t) >= 2]



@lru_cache(maxsize=4)
def _build_bm25_stats(chunks_jsonl_path: str = "data/processed/chunks/chunks.jsonl") -> dict:
    rows = load_sparse_chunks(chunks_jsonl_path)

    doc_freq = Counter()
    doc_lens: list[int] = []

    for row in rows:
        token_counts = row["_token_counts"]
        doc_lens.append(sum(token_counts.values()))

        for token in token_counts.keys():
            doc_freq[token] += 1

    n_docs = len(rows)
    avgdl = (sum(doc_lens) / n_docs) if n_docs > 0 else 0.0

    return {
        "rows": rows,
        "doc_freq": doc_freq,
        "doc_lens": doc_lens,
        "n_docs": n_docs,
        "avgdl": avgdl,
    }


def search_bm25(
    query: str,
    k: int = 5,
    chunks_jsonl_path: str = "data/processed/chunks/chunks.jsonl",
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    stats = _build_bm25_stats(chunks_jsonl_path)
    rows = stats["rows"]
    doc_freq = stats["doc_freq"]
    doc_lens = stats["doc_lens"]
    n_docs = stats["n_docs"]
    avgdl = stats["avgdl"]

    if n_docs == 0:
        return []

    query_counts = Counter(query_tokens)
    hits: list[dict] = []

    for idx, row in enumerate(rows):
        token_counts = row["_token_counts"]
        dl = doc_lens[idx]
        score = 0.0

        for token, q_count in query_counts.items():
            tf = token_counts.get(token, 0)
            if tf <= 0:
                continue

            df = doc_freq.get(token, 0)
            if df <= 0:
                continue

            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

            norm = 1.0 - b + b * (dl / avgdl) if avgdl > 0 else 1.0
            denom = tf + k1 * norm
            term_score = idf * ((tf * (k1 + 1.0)) / denom)

            score += term_score * q_count

        if score <= 0.0:
            continue

        hit = {k_: v for k_, v in row.items() if k_ != "_token_counts"}
        hit["score"] = float(score)
        hits.append(hit)

    hits.sort(
        key=lambda x: (
            -float(x.get("score", 0.0) or 0.0),
            to_pos_int(x.get("page_start")) or 10**9,
            str(x.get("chunk_id") or ""),
        )
    )

    return hits[:k]