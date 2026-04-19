from __future__ import annotations

from functools import lru_cache
from urllib.parse import quote

import boto3

from src.settings import get_settings  # pyright: ignore[reportMissingImports]


def _append_pdf_fragment(url: str, page: int | None) -> str:
    if page:
        return f"{url}#page={page}&zoom=page-fit"
    return f"{url}#zoom=page-fit"


@lru_cache(maxsize=1)
def _get_yandex_s3_client():
    s = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=s.YANDEX_STORAGE_ENDPOINT,
        aws_access_key_id=s.YANDEX_STORAGE_ACCESS_KEY,
        aws_secret_access_key=s.YANDEX_STORAGE_SECRET_KEY,
        region_name=s.YANDEX_STORAGE_REGION,
    )


def _build_yandex_source_url(ch: dict, page: int | None) -> str:
    s = get_settings()
    if not s.YANDEX_SOURCE_LINKS_ENABLED:
        return ""

    bucket = str(s.YANDEX_STORAGE_BUCKET or "").strip()
    if not bucket:
        return ""

    if s.YANDEX_SOURCE_LINKS_USE_FILENAME_AS_KEY:
        object_key = str(ch.get("filename") or "").strip()
    else:
        object_key = str(ch.get("s3_key") or "").strip()

    if not object_key:
        return ""

    try:
        client = _get_yandex_s3_client()
        url = client.generate_presigned_url(
            ClientMethod="get_object",
            Params={
                "Bucket": bucket,
                "Key": object_key,
                "ResponseContentType": "application/pdf",
                "ResponseContentDisposition": "inline",
            },
            ExpiresIn=s.YANDEX_SOURCE_LINKS_EXPIRES_SEC,
        )
        return _append_pdf_fragment(url, page)
    except Exception:
        return ""


def build_source_url(ch: dict, page: int | None) -> str:
    yandex_url = _build_yandex_source_url(ch, page)
    if yandex_url:
        return yandex_url

    s3_bucket = str(ch.get("s3_bucket") or "").strip()
    s3_key = str(ch.get("s3_key") or "").strip()

    if s3_bucket and s3_key:
        s = get_settings()
        base = (s.S3_PUBLIC_BASE_URL or s.S3_ENDPOINT).rstrip("/")
        url = f"{base}/{s3_bucket}/{quote(s3_key, safe='/')}"
        return _append_pdf_fragment(url, page)

    src = str(ch.get("source") or "").strip()
    return src