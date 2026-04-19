"""запускает загрузку чанков в индекс:
передаёт путь к chunks.jsonl и параметры батчевой обработки
в ingestion-модуль,
который добавляет подготовленные чанки в систему хранения и поиска."""


from __future__ import annotations

from rich.console import Console

from src.rag.naive_rag.ingestion import run_ingest_chunks # pyright: ignore[reportMissingImports]

console = Console()


def run(
    chunks_jsonl_path: str,
    registry_path: str = "data/corpus/sources.jsonl",
    batch_size: int = 16,
) -> int:
    return run_ingest_chunks(
        chunks_jsonl_path=chunks_jsonl_path,
        registry_path=registry_path,
        batch_size=batch_size,
    )