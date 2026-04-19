"""проверяет работу embedding-модели:
отправляет тестовый текст в embeddings api,
получает векторное представление,
и выводит результат проверки, размерность вектора и его первые значения."""


from __future__ import annotations

import httpx

from rich.console import Console
from src.settings import get_settings # pyright: ignore[reportMissingImports]

console = Console()


def run(text: str) -> int:
    s = get_settings()

    payload = {
        "model": s.EMBED_MODEL,
        "input": text,
    }

    try:
        r = httpx.post(
            f"{s.EMBED_BASE_URL}/embeddings",
            headers={"Authorization": f"Bearer {s.EMBED_API_KEY}"},
            json=payload,
            timeout=60.0,
        )
        r.raise_for_status()
        data = r.json()
        vec = data["data"][0]["embedding"]
        dim = len(vec)

        console.print("[green]Embeddings OK[/green]")
        console.print({"model": s.EMBED_MODEL, "dim": dim, "head": vec[:5]})
        return 0

    except Exception as e:
        console.print("[red]Embeddings test failed[/red]", str(e))
        return 1

def cli(argv: list[str]) -> int:
    if len(argv) < 1:
        console.print('[yellow]Usage:[/yellow] python -m src.apps.cli.main embed-test "text"')
        return 1
    return run(" ".join(argv).strip())
