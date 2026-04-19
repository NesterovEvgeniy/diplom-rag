"""запускает naive rag retrieval из cli:
принимает поисковый запрос из консоли, вызывает поиск по чанкам,
и выводит top-k найденных результатов с их score, doc_id, title и текстом."""


from __future__ import annotations

from rich.console import Console
from src.rag.naive_rag.retrieval import retrieve # pyright: ignore[reportMissingImports]
from src.rag.common.text_utils import safe_console_text # pyright: ignore[reportMissingImports]

console = Console()

'''
def _safe_console_text(value: object) -> str:
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    return str(value).encode(enc, errors="replace").decode(enc, errors="replace")
'''
def run(query: str, k: int = 5) -> int:

    hits = retrieve(query, k=k)

    console.print(f"[bold]Query:[/bold] {query}")
    console.print(f"[bold]Top-{k} hits:[/bold]")

    for idx, hit in enumerate(hits, start=1):
        text = hit.get("text", "")
        doc_id = hit.get("doc_id", "")
        title = hit.get("title") or hit.get("filename") or ""
        chunk_id = hit.get("chunk_id", "")
        score = float(hit.get("score", 0.0) or 0.0)

        console.print(f"\n{idx}) score={score:.4f} chunk={chunk_id}", markup=False, highlight=False)
        console.print(f"doc_id={safe_console_text(doc_id)}", markup=False, highlight=False)
        console.print(f"title={safe_console_text(title)}", markup=False, highlight=False)
        console.print(text, markup=False, highlight=False)

    return 0


def cli(argv: list[str]) -> int:

    if not argv:
        console.print('[yellow]Usage:[/yellow] python -m src.apps.cli.main search "query" --k 5')
        return 1

    k = 5

    if "--k" in argv:
        i = argv.index("--k")
        if i + 1 < len(argv):
            k = int(argv[i + 1])
            argv = argv[:i] + argv[i + 2 :]

    query = " ".join(argv).strip()
    return run(query, k=k)
