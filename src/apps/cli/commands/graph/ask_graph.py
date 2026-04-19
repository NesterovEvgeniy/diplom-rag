"""запускает graph rag из cli:
принимает вопрос из консоли, вызывает graph_rag pipeline,
выводит найденный ответ, количество извлечённых чанков
и список источников со страницами и ссылками."""


from __future__ import annotations

from rich.console import Console

from src.rag.graph_rag.pipeline import ask_structured  # pyright: ignore[reportMissingImports]
from src.rag.common.text_utils import safe_console_text  # pyright: ignore[reportMissingImports]


console = Console()


def run(question: str, k: int = 5) -> int:
    data = ask_structured(question, k=k)

    answer = safe_console_text(str(data.get("answer") or ""))
    sources = data.get("sources") or []
    retrieved = int(data.get("retrieved") or 0)

    console.print(f"[bold]Question:[/bold] {question}")
    console.print(f"[bold]Retrieved:[/bold] {retrieved} chunks\n")

    console.print("[bold green]Answer:[/bold green]")
    console.print(answer)
    console.print()

    console.print("[bold]Sources:[/bold]")
    for src in sources:
        n = src.get("n")
        filename = safe_console_text(str(src.get("filename") or ""))
        title = safe_console_text(str(src.get("title") or ""))
        page_start = src.get("page_start")
        page_end = src.get("page_end")
        url = safe_console_text(str(src.get("url") or ""))

        if page_start == page_end:
            pages = str(page_start) if page_start is not None else "?"
        else:
            pages = f"{page_start}-{page_end}"

        console.print(
            f"[{n}] filename={filename} | title={title} | pages={pages} | url={url}"
        )

    return 0


def cli(argv: list[str]) -> int:
    if not argv:
        console.print('[yellow]Usage:[/yellow] python -m src.apps.cli.main ask_graph "question" --k 5')
        return 1

    k = 5
    if "--k" in argv:
        i = argv.index("--k")
        if i + 1 < len(argv):
            k = int(argv[i + 1])
            argv = argv[:i] + argv[i + 2 :]

    question = " ".join(argv).strip()
    return run(question, k=k)