"""запускает graph rag retrieval из cli:
принимает поисковый запрос из консоли, вызывает graph-поиск,
и выводит top-k найденных чанков с их score, doc_id, title и текстом."""


from __future__ import annotations

from rich.console import Console

from src.rag.graph_rag.retrieval import retrieve_graph  # pyright: ignore[reportMissingImports]
from src.rag.common.text_utils import safe_console_text  # pyright: ignore[reportMissingImports]


console = Console()


def run(query: str, k: int = 5) -> int:
    hits = retrieve_graph(query, k_total=k)

    console.print(f"[bold]Query:[/bold] {query}")
    console.print(f"[bold]Top-{len(hits)} hits:[/bold]\n")

    for i, hit in enumerate(hits, start=1):
        score = float(hit.get("score") or 0.0)
        chunk_id = safe_console_text(str(hit.get("chunk_id") or ""))
        doc_id = safe_console_text(str(hit.get("doc_id") or ""))
        title = safe_console_text(str(hit.get("title") or ""))
        text = safe_console_text(str(hit.get("text") or "")).strip()

        console.print(f"{i}) score={score:.4f} chunk={chunk_id}")
        console.print(f"doc_id={doc_id}")
        console.print(f"title={title}")
        console.print(text)
        console.print()

    return 0


def cli(argv: list[str]) -> int:
    if not argv:
        console.print('[yellow]Usage:[/yellow] python -m src.apps.cli.main search_graph "query" --k 5')
        return 1

    k = 5
    if "--k" in argv:
        i = argv.index("--k")
        if i + 1 < len(argv):
            k = int(argv[i + 1])
            argv = argv[:i] + argv[i + 2 :]

    query = " ".join(argv).strip()
    return run(query, k=k)