"""содержит вспомогательные функции для распознавания отказа в ответе:
нормализует текст ответа,
проверяет, означает ли он отсутствие информации в источниках,
и помогает единообразно обрабатывать такие случаи в проекте."""


from __future__ import annotations

import re

REFUSAL_TEXT = "в источниках нет ответа"

def normalize_for_compare(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .!?:;\"'«»()[]{}")
    return text

def is_refusal_answer(answer: str) -> bool:
    text = normalize_for_compare(answer)
    return REFUSAL_TEXT in text