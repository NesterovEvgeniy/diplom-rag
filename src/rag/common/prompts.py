"""загружает текстовые prompt-шаблоны проекта:
находит нужный prompt-файл в папке configs/prompts,
читает его содержимое
и возвращает текст для использования в rag-логике."""


from __future__ import annotations

from pathlib import Path


_project_root = Path(__file__).resolve().parents[3]
_prompts_dir = _project_root / "configs" / "prompts"


def load_prompt(name: str) -> str:
    path = _prompts_dir / name
    return path.read_text(encoding="utf-8").strip()