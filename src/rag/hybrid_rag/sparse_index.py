"""подготавливает и обслуживает sparse-индекс для hybrid rag:
загружает чанки из jsonl, токенизирует их текст, собирает sparse-представление,
обогащает метаданные источников
и предоставляет функцию поиска по чанкам через bm25."""


from __future__ import annotations

import json
import re

from collections import Counter
from functools import lru_cache
from pathlib import Path

from src.rag.naive_rag.ingestion import (  # pyright: ignore[reportMissingImports]
    _normalize_filename_key,
    load_registry_meta,
)
from src.rag.common.text_utils import to_pos_int  # pyright: ignore[reportMissingImports]

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")


def _tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    return [t for t in _TOKEN_RE.findall(text) if len(t) >= 2]


@lru_cache(maxsize=4)
def load_sparse_chunks(chunks_jsonl_path: str = "data/processed/chunks/chunks.jsonl") -> list[dict]:
    path = Path(chunks_jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    registry = load_registry_meta(Path("data/corpus/sources.jsonl"))
    rows: list[dict] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSONL in chunks file at line {line_no}: {e}") from e

            text = str(row.get("text") or "").strip()
            if not text:
                continue

            filename = str(row.get("source_pdf") or row.get("filename") or "").strip()
            title = str(row.get("title") or "").strip() or Path(filename).stem
            doc_id = str(row.get("doc_id") or "").strip() or title
            chunk_id = str(row.get("chunk_id") or "").strip()

            page_num = to_pos_int(row.get("page_num"))
            page_start = to_pos_int(row.get("page_start")) or page_num
            page_end = to_pos_int(row.get("page_end")) or page_num

            token_counts = Counter(_tokenize(text))

            reg = registry.get(_normalize_filename_key(filename))

            s3_bucket = str(row.get("s3_bucket") or "")
            s3_key = str(row.get("s3_key") or "")
            source = str(row.get("source") or filename)

            if reg:
                s3_bucket = str(reg.get("s3_bucket") or s3_bucket)
                s3_key = str(reg.get("s3_key") or s3_key)
                source = str(reg.get("source") or source)

            rows.append(
                {
                    "score": 0.0,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": text,
                    "title": title,
                    "filename": filename,
                    "source": source,
                    "s3_bucket": s3_bucket,
                    "s3_key": s3_key,
                    "page_start": page_start,
                    "page_end": page_end,
                    "_token_counts": token_counts,
                }
            )

    return rows


def search_sparse(
    query: str,
    k: int = 5,
    chunks_jsonl_path: str = "data/processed/chunks/chunks.jsonl",
) -> list[dict]:
    from src.rag.hybrid_rag.bm25 import search_bm25  # pyright: ignore[reportMissingImports]

    return search_bm25(
        query,
        k=k,
        chunks_jsonl_path=chunks_jsonl_path,
    )