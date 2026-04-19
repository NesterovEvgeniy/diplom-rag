"""сохраняет manifest текущего запуска проекта:
собирает основные параметры окружения и моделей,
формирует json с конфигурацией run,
и загружает его в s3/minio bucket артефактов."""


from __future__ import annotations

import json
import boto3

from datetime import datetime, timezone
from rich.console import Console
from src.settings import get_settings # pyright: ignore[reportMissingImports]

console = Console()


def run(run_name: str = "dev") -> int:
    s = get_settings()

    manifest = {
        "run_name": run_name,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "qdrant": {"url": s.QDRANT_URL, "collection": s.QDRANT_COLLECTION},
        "llm": {"base_url": s.LLM_BASE_URL, "model": s.LLM_MODEL},
        "embed": {"base_url": s.EMBED_BASE_URL, "model": s.EMBED_MODEL, "dim": s.EMBED_DIM},
        "rag_mode": getattr(s, "RAG_MODE", "naive"),
    }

    key = f"manifests/{run_name}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    body = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=s.S3_ENDPOINT,
            aws_access_key_id=s.S3_ACCESS_KEY,
            aws_secret_access_key=s.S3_SECRET_KEY,
            region_name=s.S3_REGION,
        )

        s3.put_object(
            Bucket=s.S3_BUCKET_ARTIFACTS,
            Key=key,
            Body=body,
            ContentType="application/json; charset=utf-8",
        )

        console.print("[green]Saved manifest[/green]", {"bucket": s.S3_BUCKET_ARTIFACTS, "key": key})
        return 0
    except Exception as e:
        console.print("[red]Save manifest failed[/red]", str(e))
        return 1
