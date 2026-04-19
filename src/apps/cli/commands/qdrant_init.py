"""инициализирует коллекцию qdrant для проекта:
проверяет существование коллекции для хранения чанков,
при необходимости создаёт её заново
с параметрами dense-векторов и cosine distance."""


from __future__ import annotations

from rich.console import Console
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from src.settings import get_settings

console = Console()


def run(recreate: bool = False) -> int:
    s = get_settings()

    try:
        client = QdrantClient(url=s.QDRANT_URL, api_key=s.QDRANT_API_KEY or None)

        name = s.QDRANT_COLLECTION

        exists = client.collection_exists(name)
        if exists and recreate:
            client.delete_collection(name)
            console.print(f"[yellow]Deleted collection[/yellow]: {name}")
            exists = False

        if not exists:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=s.EMBED_DIM, distance=Distance.COSINE),
            )
            console.print(f"[green]Created collection[/green]: {name}")
        else:
            console.print(f"[green]Collection exists[/green]: {name}")

        return 0
    except Exception as e:
        console.print("[red]Qdrant init failed[/red]", str(e))
        return 1
