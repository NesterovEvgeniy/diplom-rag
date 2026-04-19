"""содержит функцию для работы с языковой моделью:
отправляет список сообщений в chat completions api,
получает текстовый ответ модели
и возвращает его для дальнейшего использования в проекте."""


from __future__ import annotations

from src.settings import get_settings # pyright: ignore[reportMissingImports]
import httpx

def chat_completion(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 350,
) -> str:
    s = get_settings()

    payload = {
        "model": model or s.LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    r = httpx.post(
        f"{s.LLM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {s.LLM_API_KEY}"},
        json=payload,
        timeout=180.0,
    )
    r.raise_for_status()

    return r.json()["choices"][0]["message"]["content"]