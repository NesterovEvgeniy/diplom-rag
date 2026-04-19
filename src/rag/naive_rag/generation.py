"""выполняет генерацию ответа через llm для rag:
собирает system и user prompt на основе вопроса и контекста чанков,
при необходимости усиливает требование к citations и русскому языку,
а затем запрашивает у модели итоговый ответ."""


from __future__ import annotations

from src.rag.common.llm import chat_completion  # pyright: ignore[reportMissingImports]
from src.rag.common.prompts import load_prompt  # pyright: ignore[reportMissingImports]
from src.settings import get_settings  # pyright: ignore[reportMissingImports]


def llm_answer(
    question: str,
    context: str,
    n_sources: int,
    force_citations: bool = False,
    force_russian: bool = False,
) -> str:
    s = get_settings()

    system_parts: list[str] = [
        load_prompt("rag_system.txt").format(n_sources=n_sources)
    ]

    if force_citations:
        system_parts.append(load_prompt("rag_force_citations.txt"))

    if force_russian:
        system_parts.append("LANGUAGE RULE: answer strictly in Russian only.")

    system = "\n\n".join(part.strip() for part in system_parts if part.strip())

    user = f"ВОПРОС:\n{question}\n\nИСТОЧНИКИ:\n{context}\n\nОТВЕТ:"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    return chat_completion(
        messages=messages,
        model=s.LLM_MODEL,
        temperature=0.0,
        max_tokens=350,
    )