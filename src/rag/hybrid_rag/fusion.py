"""объединяет результаты dense и sparse retrieval:
сливает найденные чанки из двух поисков,
пересчитывает их общий score по методу reciprocal rank fusion (rrf),
и возвращает итоговый top-k список для hybrid rag."""


from __future__ import annotations


def _chunk_key(ch: dict) -> tuple[str, str, str]:
    return (
        str(ch.get("doc_id") or ""),
        str(ch.get("chunk_id") or ""),
        str(ch.get("filename") or ""),
    )


def _prefer_dense_metadata(current: dict, dense_hit: dict) -> dict:
    merged = dict(current)

    for key in (
        "text",
        "title",
        "filename",
        "source",
        "s3_bucket",
        "s3_key",
        "page_start",
        "page_end",
        "doc_id",
        "chunk_id",
    ):
        value = dense_hit.get(key)
        if value not in (None, ""):
            merged[key] = value

    return merged


def fuse_rrf(
    dense_hits: list[dict],
    sparse_hits: list[dict],
    k: int,
    rrf_k: int = 60,
) -> list[dict]:
    merged: dict[tuple[str, str, str], dict] = {}

    for rank, ch in enumerate(dense_hits, start=1):
        key = _chunk_key(ch)
        score_add = 1.0 / (rrf_k + rank)

        if key not in merged:
            item = dict(ch)
            item["score"] = 0.0
            item["dense_score"] = float(ch.get("score", 0.0) or 0.0)
            item["sparse_score"] = 0.0
            item["rrf_score"] = 0.0
            merged[key] = item
        else:
            merged[key] = _prefer_dense_metadata(merged[key], ch)
            merged[key]["dense_score"] = float(ch.get("score", 0.0) or 0.0)

        merged[key]["rrf_score"] += score_add
        merged[key]["score"] = merged[key]["rrf_score"]

    for rank, ch in enumerate(sparse_hits, start=1):
        key = _chunk_key(ch)
        score_add = 1.0 / (rrf_k + rank)

        if key not in merged:
            item = dict(ch)
            item["score"] = 0.0
            item["dense_score"] = 0.0
            item["sparse_score"] = float(ch.get("score", 0.0) or 0.0)
            item["rrf_score"] = 0.0
            merged[key] = item
        else:
            merged[key]["sparse_score"] = float(ch.get("score", 0.0) or 0.0)

        merged[key]["rrf_score"] += score_add
        merged[key]["score"] = merged[key]["rrf_score"]

    hits = list(merged.values())
    hits.sort(key=lambda x: -float(x.get("rrf_score", 0.0) or 0.0))
    return hits[:k]