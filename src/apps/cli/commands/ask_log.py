"""запускает naive rag с сохранением лога:
принимает вопрос из консоли, выполняет retrieval и generation,
собирает ответ вместе с найденными чанками и конфигурацией,
и сохраняет результат в jsonl-файл."""


from __future__ import annotations

from pathlib import Path

from rich.console import Console

from src.rag.common.logging import append_jsonl_record, build_common_config, now_utc_iso # pyright: ignore[reportMissingImports]
from src.rag.naive_rag.context import build_context # pyright: ignore[reportMissingImports]
from src.rag.naive_rag.generation import llm_answer # pyright: ignore[reportMissingImports]
from src.rag.naive_rag.retrieval import retrieve # pyright: ignore[reportMissingImports]

console = Console()


def run(question: str, k: int = 5, out_path: str = "data/manual_runs/answers.jsonl") -> int:
    chunks = retrieve(question, k=k)
    context = build_context(chunks) if chunks else ""
    answer = llm_answer(question, context, n_sources=len(chunks)) if chunks else "В источниках нет ответа."

    record = {
        "ts_utc": now_utc_iso(),
        "question": question,
        "k": k,
        "answer": answer,
        "chunks": chunks,
        "config": build_common_config(),
    }

    p = append_jsonl_record(out_path, record)
    console.print("[green]Saved run[/green]", {"file": str(p), "question": question})
    return 0


def cli(argv: list[str]) -> int:
    if not argv:
        console.print('[yellow]Usage:[/yellow] python -m src.apps.cli.main ask-log "question" --k 5')
        return 1

    k = 5
    if "--k" in argv:
        i = argv.index("--k")
        if i + 1 < len(argv):
            k = int(argv[i + 1])
            argv = argv[:i] + argv[i + 2 :]

    question = " ".join(argv).strip()
    return run(question, k=k)
