"""содержит общие вспомогательные функции для работы с текстом и числами:
безопасно подготавливает текст для вывода в консоль,
а также преобразует значения в положительные целые числа
для использования в cli и внутренней логике проекта."""


from __future__ import annotations

import sys


def safe_console_text(value: object) -> str:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    return str(value).encode(enc, errors="replace").decode(enc, errors="replace")


def to_pos_int(value: object) -> int | None:
    try:
        num = int(value)
    except Exception:
        return None
    return num if num > 0 else None