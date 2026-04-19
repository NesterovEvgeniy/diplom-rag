"""запускает naive rag retrieval с сохранением лога:
принимает поисковый запрос из консоли, выполняет поиск по чанкам,
сохраняет найденные результаты и конфигурацию запуска
в jsonl-файл для последующего анализа."""


from __future__ import annotations

from pathlib import Path

from rich.console import Console

from src.rag.common.logging import append_jsonl_record, build_common_config, now_utc_iso # pyright: ignore[reportMissingImports]
from src.rag.naive_rag.retrieval import retrieve # pyright: ignore[reportMissingImports]

console = Console()


def run(query: str, k: int = 5, out_path: str = "data/manual_runs/search.jsonl") -> int:
    chunks = retrieve(query, k=k)

    record = {
        "ts_utc": now_utc_iso(),
        "query": query,
        "k": k,
        "chunks": chunks,
        "config": build_common_config(),
    }

    p = append_jsonl_record(out_path, record)
    console.print("[green]Saved search run[/green]", {"file": str(p), "query": query})
    return 0


def cli(argv: list[str]) -> int:
    if not argv:
        console.print('[yellow]Usage:[/yellow] python -m src.apps.cli.main search-log "query" --k 5')
        return 1

    k = 5
    if "--k" in argv:
        i = argv.index("--k")
        if i + 1 < len(argv):
            k = int(argv[i + 1])
            argv = argv[:i] + argv[i + 2 :]

    query = " ".join(argv).strip()
    return run(query, k=k)