"""выполняет ingestion подготовленных чанков в qdrant:
читает chunks.jsonl, группирует чанки по документам, получает их embeddings,
обогащает метаданные из registry
и загружает чанки в векторную коллекцию для последующего retrieval."""


from __future__ import annotations

import hashlib
import json
import re
import unicodedata

from pathlib import Path
from typing import Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.models import PointStruct
from rich.console import Console
from src.rag.common.embeddings import embed_batch # pyright: ignore[reportMissingImports]
from src.rag.common.text_utils import safe_console_text, to_pos_int # pyright: ignore[reportMissingImports]
from src.settings import get_settings # pyright: ignore[reportMissingImports]

console = Console()


def _normalize_filename_key(value: str) -> str:
    value = Path(str(value)).name
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.lower().strip()
    value = re.sub(r"\.pdf$", "", value)
    value = value.replace("ё", "е")
    value = re.sub(r"[^a-zа-я0-9]+", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def stable_chunk_point_id(filename: str, chunk_id: str) -> int:
    digest = hashlib.sha1(f"{filename}:{chunk_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def load_registry_meta(path: Path) -> dict[str, dict[str, str]]:
    meta_by_filename: dict[str, dict[str, str]] = {}

    if not path.exists():
        return meta_by_filename

    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            line = line.lstrip("\ufeff")

            try:
                rec = json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON in registry at line {line_no}: {e}") from e

            filename = str(rec.get("filename") or rec.get("file") or "").strip()
            if not filename:
                continue

            key = _normalize_filename_key(filename)

            meta_by_filename[key] = {
                "filename": Path(filename).name,
                "registry_doc_id": str(rec.get("doc_id") or "").strip(),
                "s3_bucket": str(rec.get("s3_bucket") or rec.get("bucket") or "").strip(),
                "s3_key": str(rec.get("s3_key") or rec.get("key") or "").strip(),
                "source": str(rec.get("source") or "").strip(),
            }

    return meta_by_filename


def load_chunks_grouped(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    groups: dict[str, list[dict[str, Any]]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSONL in chunks file at line {line_no}: {e}") from e

            filename = str(row.get("source_pdf") or row.get("filename") or "").strip()
            if not filename:
                raise ValueError(f"Missing source_pdf/filename in chunks file at line {line_no}")

            filename = Path(filename).name
            groups.setdefault(filename, []).append(row)

    for filename, rows in groups.items():
        rows.sort(
            key=lambda r: (
                to_pos_int(r.get("page_start") or r.get("page_num")) or 0,
                to_pos_int(r.get("chunk_index")) or 0,
                str(r.get("chunk_id") or ""),
            )
        )

    return groups


def delete_existing_doc_points(qdrant: QdrantClient, collection_name: str, filename: str) -> None:
    qdrant.delete(
        collection_name=collection_name,
        points_selector=qm.FilterSelector(
            filter=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="filename",
                        match=qm.MatchValue(value=filename),
                    )
                ]
            )
        ),
    )


def run_ingest_chunks(
    chunks_jsonl_path: str,
    registry_path: str = "data/corpus/sources.jsonl",
    batch_size: int = 16,
) -> int:
    s = get_settings()

    chunks_path = Path(chunks_jsonl_path)
    registry = load_registry_meta(Path(registry_path))
    grouped_chunks = load_chunks_grouped(chunks_path)

    if not grouped_chunks:
        console.print("[yellow]Chunks file is empty[/yellow]", str(chunks_path))
        return 0

    qdrant = QdrantClient(url=s.QDRANT_URL, api_key=s.QDRANT_API_KEY or None)

    total_points = 0
    total_docs = 0

    for filename, rows in grouped_chunks.items():
        title_fallback = Path(filename).stem
        reg = registry.get(_normalize_filename_key(filename))

        console.print(
            "[bold]Indexing[/bold]",
            {
                "file": safe_console_text(filename),
                "chunks": len(rows),
                "registry_match": bool(reg),
            },
        )

        delete_existing_doc_points(qdrant, s.QDRANT_COLLECTION, filename)

        doc_points = 0

        for b in range(0, len(rows), batch_size):
            batch_rows_raw = rows[b : b + batch_size]

            batch_rows: list[dict[str, Any]] = []
            batch_texts: list[str] = []

            for row in batch_rows_raw:
                text = str(row.get("text") or "").strip()
                if not text:
                    continue
                batch_rows.append(row)
                batch_texts.append(text)

            if not batch_rows:
                continue

            vectors = embed_batch(batch_texts)
            points: list[PointStruct] = []

            for row, vec in zip(batch_rows, vectors):
                doc_id = str(row.get("doc_id") or "").strip() or title_fallback
                title = str(row.get("title") or "").strip() or title_fallback
                chunk_id = str(row.get("chunk_id") or "").strip()

                if not chunk_id:
                    page_num_tmp = to_pos_int(row.get("page_num")) or 0
                    chunk_index_tmp = to_pos_int(row.get("chunk_index")) or 0
                    chunk_id = f"{doc_id}_p{page_num_tmp:04d}_c{chunk_index_tmp:03d}"

                page_num = to_pos_int(row.get("page_num"))
                page_start = to_pos_int(row.get("page_start")) or page_num
                page_end = to_pos_int(row.get("page_end")) or page_num
                chunk_index = to_pos_int(row.get("chunk_index"))

                point_id = stable_chunk_point_id(filename, chunk_id)

                payload: dict[str, Any] = {
                    "doc_id": doc_id,
                    "title": title,
                    "filename": filename,
                    "source_pdf": filename,
                    "source": filename,
                    "chunk_id": chunk_id,
                    "text": str(row.get("text") or "").strip(),
                }

                if page_num is not None:
                    payload["page_num"] = page_num
                if page_start is not None:
                    payload["page_start"] = page_start
                if page_end is not None:
                    payload["page_end"] = page_end
                if chunk_index is not None:
                    payload["chunk_index"] = chunk_index

                if reg:
                    if reg.get("registry_doc_id"):
                        payload["registry_doc_id"] = reg["registry_doc_id"]
                    if reg.get("s3_bucket"):
                        payload["s3_bucket"] = reg["s3_bucket"]
                    if reg.get("s3_key"):
                        payload["s3_key"] = reg["s3_key"]
                    if reg.get("source"):
                        payload["source"] = reg["source"]

                points.append(PointStruct(id=point_id, vector=vec, payload=payload))

            if points:
                qdrant.upsert(collection_name=s.QDRANT_COLLECTION, points=points)
                doc_points += len(points)

        total_points += doc_points
        total_docs += 1

        console.print(
            f"[green]Indexed[/green] file={safe_console_text(filename)} chunks={doc_points}"
        )

    console.print(
        f"[bold green]Done[/bold green] docs={total_docs} total_points={total_points}"
    )
    return 0