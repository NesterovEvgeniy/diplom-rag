"""инициализирует s3/minio-хранилище проекта:
проверяет наличие нужных bucket'ов, при необходимости создаёт их,
и настраивает публичный доступ на чтение для bucket с исходными источниками."""


from __future__ import annotations

import json
import boto3

from rich.console import Console
from src.settings import get_settings # pyright: ignore[reportMissingImports]

console = Console()


def ensure_bucket(s3, bucket_name: str) -> None:
    
    try:

        s3.head_bucket(Bucket=bucket_name)
        console.print(f"[green]Bucket exists[/green]: {bucket_name}")
    except Exception:

        s3.create_bucket(Bucket=bucket_name)
        console.print(f"[green]Bucket created[/green]: {bucket_name}")


def run() -> int:

    s = get_settings()

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=s.S3_ENDPOINT,
            aws_access_key_id=s.S3_ACCESS_KEY,
            aws_secret_access_key=s.S3_SECRET_KEY,
            region_name=s.S3_REGION,
        )

        ensure_bucket(s3, s.S3_BUCKET_SOURCES)
        ensure_bucket(s3, s.S3_BUCKET_ARTIFACTS)

        public_read_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "PublicReadObjects",
                    "Effect": "Allow",
                    "Principal": {"AWS": ["*"]},
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{s.S3_BUCKET_SOURCES}/*"],
                }
            ],
        }
        s3.put_bucket_policy(Bucket=s.S3_BUCKET_SOURCES, Policy=json.dumps(public_read_policy))
        console.print(f"[green]Bucket policy set[/green]: {s.S3_BUCKET_SOURCES} public-read")

        return 0
    except Exception as e:
        console.print("[red]MinIO init failed[/red]", str(e))
        return 1
