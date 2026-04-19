"""реализует dense retrieval для naive rag:
преобразует запрос в embedding-вектор, ищет ближайшие чанки в qdrant,
извлекает их payload и score
и возвращает список релевантных результатов для дальнейшей генерации ответа."""


from __future__ import annotations

from qdrant_client import QdrantClient
from src.rag.common.embeddings import embed_text # pyright: ignore[reportMissingImports]
from src.settings import get_settings # pyright: ignore[reportMissingImports]

def retrieve(query: str, k: int) -> list[dict]:
    s = get_settings()
    client = QdrantClient(url=s.QDRANT_URL, api_key=s.QDRANT_API_KEY or None)

    vec = embed_text(query)
    result = client.query_points(
        collection_name=s.QDRANT_COLLECTION,
        query=vec,
        limit=k,
        with_payload=[
            "text",
            "title",
            "filename",
            "chunk_id",
            "doc_id",
            "source",
            "s3_key",
            "s3_bucket",
            "page_start",
            "page_end",
        ],
    )

    chunks: list[dict] = []
    for p in result.points:
        payload = p.payload or {}
        chunks.append(
            {
                "score": float(getattr(p, "score", 0.0) or 0.0),
                "doc_id": "" if payload.get("doc_id") is None else str(payload.get("doc_id")),
                "chunk_id": "" if payload.get("chunk_id") is None else str(payload.get("chunk_id")),
                "text": payload.get("text", "") or "",
                "title": str(payload.get("title") or ""),
                "filename": str(payload.get("filename") or ""),
                "source": payload.get("source", "") or "",
                "s3_bucket": payload.get("s3_bucket", "") or "",
                "s3_key": payload.get("s3_key", "") or "",
                "page_start": payload.get("page_start"),
                "page_end": payload.get("page_end"),
            }
        )

    return chunks