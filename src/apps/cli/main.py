"""запускает основной cli-интерфейс проекта:
принимает команду из консоли, перенаправляет её в нужный модуль,
позволяет проверять сервисы, выполнять ingest, search, ask,
запускать bot и разные режимы rag."""


from __future__ import annotations

import sys
import boto3
import httpx

from rich.console import Console
from src.settings import get_settings # pyright: ignore[reportMissingImports]
from src.apps.cli.commands.graph.ask_graph import cli as ask_graph_cli # pyright: ignore[reportMissingImports]
from src.apps.cli.commands.graph.search_graph import cli as search_graph_cli # pyright: ignore[reportMissingImports]

console = Console()


def ping() -> int:

    s = get_settings()

    ok = True

    try:
        r = httpx.get(f"{s.QDRANT_URL}/", timeout=5.0)
        r.raise_for_status()
        console.print("[green]Qdrant OK[/green]", r.json())
    except Exception as e:
        ok = False
        console.print("[red]Qdrant FAIL[/red]", str(e))

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=s.S3_ENDPOINT,
            aws_access_key_id=s.S3_ACCESS_KEY,
            aws_secret_access_key=s.S3_SECRET_KEY,
            region_name=s.S3_REGION,
        )

        buckets = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
        console.print("[green]MinIO OK[/green]", {"buckets": buckets})

        for need in (s.S3_BUCKET_SOURCES, s.S3_BUCKET_ARTIFACTS):
            if need not in buckets:
                ok = False
                console.print(f"[red]MinIO bucket missing[/red] {need}")
    except Exception as e:
        ok = False
        console.print("[red]MinIO FAIL[/red]", str(e))

    try:
        r = httpx.get(f"{s.LLM_BASE_URL}/models", timeout=5.0, headers={"Authorization": f"Bearer {s.LLM_API_KEY}"})
        r.raise_for_status()

        data = r.json().get("data", [])
        model_ids = [m.get("id") for m in data if isinstance(m, dict)]

        console.print("[green]LLM OK[/green]", {"models": model_ids})

        if s.LLM_MODEL not in model_ids:
            ok = False
            console.print(f"[red]LLM model missing[/red] {s.LLM_MODEL}")
    except Exception as e:
        ok = False
        console.print("[red]LLM FAIL[/red]", str(e))
    return 0 if ok else 1


def main() -> None:

    if len(sys.argv) < 2:
        console.print("[yellow]Usage:[/yellow] python -m src.apps.cli.main ping")
        raise SystemExit(1)

    cmd = sys.argv[1].strip().lower()

    if cmd == "ping":
        raise SystemExit(ping())
    if cmd == "minio-init":
        from src.apps.cli.commands.minio_init import run
        raise SystemExit(run())
    if cmd == "llm-ask":
        from src.apps.cli.commands.llm_ask import cli
        raise SystemExit(cli(sys.argv[2:]))
    if cmd == "qdrant-init":
        from src.apps.cli.commands.qdrant_init import run
        raise SystemExit(run(recreate="--recreate" in sys.argv))
    if cmd == "embed-test":
        from src.apps.cli.commands.embed_test import cli
        raise SystemExit(cli(sys.argv[2:]))
    if cmd == "ingest-chunks":
        from src.apps.cli.commands.ingest_chunks import run
        path = sys.argv[2] if len(sys.argv) > 2 else "data/processed/chunks/chunks.jsonl"
        raise SystemExit(run(path))
    if cmd == "search":
        from src.apps.cli.commands.search import cli
        raise SystemExit(cli(sys.argv[2:]))
    if cmd == "ask":
        from src.apps.cli.commands.ask import cli
        raise SystemExit(cli(sys.argv[2:]))
    elif cmd == "ask_hybrid":
        from src.apps.cli.commands.hybrid.ask_hybrid import cli
        raise SystemExit(cli(sys.argv[2:]))
    elif cmd == "search_hybrid":
        from src.apps.cli.commands.hybrid.search_hybrid import cli
        raise SystemExit(cli(sys.argv[2:]))
    if cmd == "save-manifest":
        from src.apps.cli.commands.save_manifest import run
        name = sys.argv[2] if len(sys.argv) > 2 else "dev"
        raise SystemExit(run(name))
    if cmd == "ask-log":
        from src.apps.cli.commands.ask_log import cli
        raise SystemExit(cli(sys.argv[2:]))
    if cmd == "search-log":
        from src.apps.cli.commands.search_log import cli
        raise SystemExit(cli(sys.argv[2:]))
    if cmd == "sources-upload":
        from src.apps.cli.commands.sources_upload import run
        folder = sys.argv[2] if len(sys.argv) > 2 else ""
        raise SystemExit(run(folder))
    if cmd == "bot":
        import asyncio
        from src.apps.bot.main import run_bot
        asyncio.run(run_bot())
        raise SystemExit(0)
    if cmd == "ask_graph":
        from src.apps.cli.commands.graph.ask_graph import cli
        raise SystemExit(cli(sys.argv[2:]))
    if cmd == "search_graph":
        from src.apps.cli.commands.graph.search_graph import cli
        raise SystemExit(cli(sys.argv[2:]))

    console.print(f"[red]Unknown command:[/red] {cmd}")
    raise SystemExit(1)

if __name__ == "__main__":
    main()
