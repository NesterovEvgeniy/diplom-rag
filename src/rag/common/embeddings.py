"""содержит функции для получения embedding-векторов:
отправляет один текст или список текстов в embeddings api,
получает их векторные представления
и возвращает их для дальнейшего поиска и индексации."""


from __future__ import annotations

from src.settings import get_settings # pyright: ignore[reportMissingImports]
import httpx

def embed_text(text: str) -> list[float]:

    s = get_settings()
    payload = {"model": s.EMBED_MODEL, "input": text}

    r = httpx.post(
        f"{s.EMBED_BASE_URL}/embeddings",
        headers={"Authorization": f"Bearer {s.EMBED_API_KEY}"},
        json=payload,
        timeout=60.0,
    )
    r.raise_for_status()

    return r.json()["data"][0]["embedding"]


def embed_batch(texts: list[str]) -> list[list[float]]:

    s = get_settings()
    payload = {"model": s.EMBED_MODEL, "input": texts}

    r = httpx.post(
        f"{s.EMBED_BASE_URL}/embeddings",
        headers={"Authorization": f"Bearer {s.EMBED_API_KEY}"},
        json=payload,
        timeout=120.0,
    )
    r.raise_for_status()

    data = r.json()["data"]
    return [item["embedding"] for item in data]
