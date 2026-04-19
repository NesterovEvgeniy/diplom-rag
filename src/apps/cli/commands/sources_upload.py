"""загружает исходные файлы источников в s3/minio-хранилище:
вычисляет sha256 для каждого файла, пропускает уже известные,
сохраняет новые файлы в bucket sources
и записывает их метаданные в registry jsonl."""


from __future__ import annotations

import hashlib
import json
import mimetypes
import boto3

from datetime import datetime, timezone
from pathlib import Path
from rich.console import Console
from src.settings import get_settings # pyright: ignore[reportMissingImports]
from src.rag.common.text_utils import safe_console_text # pyright: ignore[reportMissingImports]

console = Console()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_known_sha256(registry_path: Path) -> set[str]:
    known: set[str] = set()
    if not registry_path.exists():
        return known

    with registry_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                s = obj.get("sha256")
                if s:
                    known.add(s)
            except Exception:
                continue
    return known


def run(folder: str, registry: str = "data/corpus/sources.jsonl") -> int:
    s = get_settings()

    src_dir = Path(folder)
    if not src_dir.exists():
        console.print("[red]Folder not found[/red]", str(src_dir))
        return 1

    reg_path = Path(registry)
    reg_path.parent.mkdir(parents=True, exist_ok=True)

    known = load_known_sha256(reg_path)

    s3 = boto3.client(
        "s3",
        endpoint_url=s.S3_ENDPOINT,
        aws_access_key_id=s.S3_ACCESS_KEY,
        aws_secret_access_key=s.S3_SECRET_KEY,
        region_name=s.S3_REGION,
    )

    uploaded = 0

    for path in sorted(src_dir.rglob("*")):
        if path.is_dir():
            continue

        file_sha = sha256_file(path)
        if file_sha in known:
            console.print("[yellow]Skip (already in registry)[/yellow]", safe_console_text(path.name))
            continue

        mime, _ = mimetypes.guess_type(str(path))
        if not mime:
            mime = "application/octet-stream"

        ext = path.suffix.lower() or ".bin"
        key = f"raw/{file_sha}{ext}"

        s3.put_object(
            Bucket=s.S3_BUCKET_SOURCES,
            Key=key,
            Body=path.read_bytes(),
            ContentType=mime,
        )

        record = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "doc_id": file_sha,
            "filename": path.name,
            "sha256": file_sha,
            "bytes": path.stat().st_size,
            "mime": mime,
            "s3_bucket": s.S3_BUCKET_SOURCES,
            "s3_key": key,
            "source": "local_upload",
        }

        with reg_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        known.add(file_sha)
        uploaded += 1
        console.print(
            "[green]Uploaded[/green]",
            {"file": safe_console_text(path.name), "key": safe_console_text(key)},
        )

    console.print(f"[bold green]Done[/bold green] uploaded={uploaded} registry={str(reg_path)}")
    return 0
