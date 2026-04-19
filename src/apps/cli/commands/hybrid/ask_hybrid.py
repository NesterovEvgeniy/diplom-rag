"""запускает hybrid rag из cli:
принимает вопрос из консоли, вызывает hybrid_rag pipeline,
выводит найденный ответ, количество извлечённых чанков
и список источников со страницами и ссылками."""


from __future__ import annotations

from src.rag.naive_rag.generation import llm_answer  # pyright: ignore[reportMissingImports]
from src.rag.hybrid_rag.pipeline import ask_structured  # pyright: ignore[reportMissingImports]
from src.rag.common.text_utils import safe_console_text  # pyright: ignore[reportMissingImports]
from rich.console import Console

console = Console()


def run(question: str, k: int = 5) -> int:
    data = ask_structured(
    question,
    k=k,
    llm_answer=llm_answer,
    )

    console.print(f"[bold]Question:[/bold] {question}")
    console.print(f"[bold]Retrieved:[/bold] {data['retrieved']} chunks")

    if data["answer"].strip() == "В источниках нет ответа.":
        console.print("\n[red]В источниках нет ответа.[/red]")
        return 0

    console.print("\n[bold green]Answer:[/bold green]")
    console.print(safe_console_text(data["answer"]), markup=False)

    print("\nSources:")
    for s in data["sources"]:
        page_start = s.get("page_start")
        page_end = s.get("page_end")

        if page_start and page_end:
            pages_text = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
        elif page_start:
            pages_text = str(page_start)
        else:
            pages_text = "n/a"

        print(
            f"[{s['n']}] "
            f"filename={s.get('filename') or 'n/a'} | "
            f"title={s.get('title') or 'n/a'} | "
            f"pages={pages_text} | "
            f"url={s.get('url') or 'n/a'}"
        )

    return 0


def cli(argv: list[str]) -> int:
    if not argv:
        console.print('[yellow]Usage:[/yellow] python -m src.apps.cli.main ask_hybrid "question" --k 5')
        return 1

    k = 5
    if "--k" in argv:
        i = argv.index("--k")
        if i + 1 < len(argv):
            k = int(argv[i + 1])
            argv = argv[:i] + argv[i + 2 :]

    question = " ".join(argv).strip()
    return run(question, k=k)