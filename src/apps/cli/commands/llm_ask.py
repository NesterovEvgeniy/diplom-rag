"""выполняет прямой запрос к языковой модели без rag:
отправляет пользовательский prompt в llm,
получает обычный ответ модели и выводит его в консоль."""


from __future__ import annotations

from rich.console import Console
from src.rag.common.llm import chat_completion # pyright: ignore[reportMissingImports]

console = Console()


def run(prompt: str) -> int:

    messages = [
        {"role": "system", "content": "Ты полезный ассистент. Отвечай кратко."},
        {"role": "user", "content": prompt},
    ]

    try:
        answer = chat_completion(
            messages=messages,
            temperature=0.2,
            max_tokens=256,
        )
    
        console.print("[green]LLM answer:[/green]")
        console.print(answer)
        return 0

    except Exception as e:
        console.print("[red]LLM ask failed[/red]", str(e))
        return 1


def cli(argv: list[str]) -> int:

    if len(argv) < 1:
        console.print('[yellow]Usage:[/yellow] python -m src.apps.cli.main llm-ask "your prompt"')
        return 1

    prompt = " ".join(argv).strip()
    return run(prompt)
