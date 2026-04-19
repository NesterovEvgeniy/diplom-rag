"""содержит общие вспомогательные функции для логирования:
создаёт utc-временную метку, собирает базовую конфигурацию запуска проекта
и сохраняет записи в jsonl-файлы
для ручных запусков и последующего анализа."""


from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.settings import get_settings # pyright: ignore[reportMissingImports]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_common_config() -> dict[str, Any]:
    s = get_settings()
    return {
        "qdrant_collection": s.QDRANT_COLLECTION,
        "llm_model": s.LLM_MODEL,
        "embed_model": s.EMBED_MODEL,
        "rag_mode": getattr(s, "RAG_MODE", "naive"),
    }


def append_jsonl_record(out_path: str, record: dict[str, Any]) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return p