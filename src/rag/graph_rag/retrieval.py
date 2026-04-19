"""реализует retrieval для graph rag:
берёт seed-чанки через dense-поиск, расширяет их соседями из графа,
обогащает метаданные чанков
и возвращает итоговый список релевантных результатов."""


from __future__ import annotations

import json

from pathlib import Path

from src.rag.naive_rag.retrieval import retrieve as dense_retrieve  # pyright: ignore[reportMissingImports]
from src.rag.graph_rag.graph_store import get_neighbors  # pyright: ignore[reportMissingImports]
from src.rag.naive_rag.ingestion import (  # pyright: ignore[reportMissingImports]
    _normalize_filename_key,
    load_registry_meta,
)


CHUNKS_PATH = Path("data/processed/chunks/chunks.jsonl")


def load_chunks_by_id(
    chunks_path: Path = CHUNKS_PATH,
    registry_path: Path = Path("data/corpus/sources.jsonl"),
) -> dict[str, dict]:
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks file not found: {chunks_path}")
    
    registry = load_registry_meta(registry_path)
    rows: dict[str, dict] = {}

    with chunks_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except Exception as e:
                raise ValueError(f"invalid json in chunks file at line {line_no}: {e}") from e

            chunk_id = str(row.get("chunk_id") or "").strip()
            if not chunk_id:
                continue
            
            filename = str(row.get("filename") or row.get("source_pdf") or "").strip()
            reg = registry.get(_normalize_filename_key(filename))

            if reg:
                row["filename"] = str(reg.get("source_filename") or filename)
                row["source"] = str(reg.get("source") or row.get("source") or "")
                row["s3_bucket"] = str(reg.get("s3_bucket") or row.get("s3_bucket") or "")
                row["s3_key"] = str(reg.get("s3_key") or row.get("s3_key") or "")
            else:
                row["filename"] = filename
                row["source"] = str(row.get("source") or "")
                row["s3_bucket"] = str(row.get("s3_bucket") or "")
                row["s3_key"] = str(row.get("s3_key") or "")


            rows[chunk_id] = row

    return rows


def _normalize_chunk_payload(row: dict) -> dict:
    return {
        "score": float(row.get("score") or 0.0),
        "doc_id": str(row.get("doc_id") or ""),
        "chunk_id": str(row.get("chunk_id") or ""),
        "text": str(row.get("text") or ""),
        "title": str(row.get("title") or ""),
        "filename": str(row.get("filename") or row.get("source_pdf") or ""),
        "source": str(row.get("source") or row.get("source_pdf") or ""),
        "s3_bucket": str(row.get("s3_bucket") or ""),
        "s3_key": str(row.get("s3_key") or ""),
        "page_start": row.get("page_start") if row.get("page_start") is not None else row.get("page_num"),
        "page_end": row.get("page_end") if row.get("page_end") is not None else row.get("page_num"),
    }


def retrieve_graph(
    query: str,
    k_total: int = 5,
    k_seed: int = 2,
    max_neighbors_per_seed: int = 2,
) -> list[dict]:
    seed_hits = dense_retrieve(query, k=max(1, min(k_seed, k_total)))
    if not seed_hits:
        return []

    chunks_by_id = load_chunks_by_id()
    results: list[dict] = []
    seen: set[str] = set()

    for hit in seed_hits:
        chunk_id = str(hit.get("chunk_id") or "").strip()
        if not chunk_id or chunk_id in seen:
            continue

        seen.add(chunk_id)
        results.append(hit)

    if len(results) >= k_total:
        return results[:k_total]

    for hit in seed_hits:
        if len(results) >= k_total:
            break

        chunk_id = str(hit.get("chunk_id") or "").strip()
        if not chunk_id:
            continue

        neighbors = get_neighbors(chunk_id)
        added_for_seed = 0

        for neighbor_id in neighbors:
            if len(results) >= k_total:
                break

            if added_for_seed >= max_neighbors_per_seed:
                break

            if neighbor_id in seen:
                continue

            row = chunks_by_id.get(neighbor_id)
            if not row:
                continue

            neighbor_hit = _normalize_chunk_payload(row)
            neighbor_hit["score"] = float(hit.get("score") or 0.0) * 0.95

            seen.add(neighbor_id)
            results.append(neighbor_hit)
            added_for_seed += 1

    return results[:k_total]