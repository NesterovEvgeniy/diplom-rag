"""реализует основной graph rag pipeline:
получает релевантные чанки через graph retrieval, собирает контекст,
запрашивает ответ у llm, проверяет и нормализует citations,
а затем возвращает структурированный результат с ответом и источниками."""


from __future__ import annotations

from src.rag.common.citations import (  # pyright: ignore[reportMissingImports]
    _assign_single_chunk_citations,
    _has_valid_citations,
    remap_citations_by_first_appearance,
)
from src.rag.common.refusals import is_refusal_answer  # pyright: ignore[reportMissingImports]
from src.rag.common.text_utils import safe_console_text, to_pos_int  # pyright: ignore[reportMissingImports]
from src.rag.naive_rag.context import build_context  # pyright: ignore[reportMissingImports]
from src.rag.naive_rag.generation import llm_answer  # pyright: ignore[reportMissingImports]
from src.rag.graph_rag.retrieval import retrieve_graph  # pyright: ignore[reportMissingImports]
from src.settings import get_settings  # pyright: ignore[reportMissingImports]
from src.storage.source_links import build_source_url  # pyright: ignore[reportMissingImports]


def ask_structured(
    question: str,
    k: int = 5,
) -> dict:
    chunks = retrieve_graph(question, k_total=k)
    if not chunks:
        return {
            "answer": "В источниках нет ответа.",
            "sources": [],
            "retrieved": 0,
            "chunks": [],
        }

    s = get_settings()
    context = build_context(chunks)

    answer = llm_answer(question, context, n_sources=len(chunks), force_citations=False)
    if answer.strip() != "В источниках нет ответа." and not _has_valid_citations(answer, n_sources=len(chunks)):
        answer = llm_answer(question, context, n_sources=len(chunks), force_citations=True)

    if is_refusal_answer(answer):
        return {
            "answer": "В источниках нет ответа.",
            "sources": [],
            "retrieved": len(chunks),
            "chunks": chunks,
        }

    if s.STRICT_CITATIONS and not _has_valid_citations(answer, n_sources=len(chunks)):
        return {
            "answer": "В источниках нет ответа.",
            "sources": [],
            "retrieved": len(chunks),
            "chunks": chunks,
        }

    answer, used_chunk_idxs = _assign_single_chunk_citations(answer, chunks)
    answer, used_chunk_idxs = remap_citations_by_first_appearance(answer, n_sources=len(chunks))

    if not used_chunk_idxs:
        used_chunk_idxs = list(range(len(chunks)))

    sources: list[dict] = []
    for i, chunk_idx in enumerate(used_chunk_idxs, start=1):
        if chunk_idx < 0 or chunk_idx >= len(chunks):
            continue

        ch = chunks[chunk_idx]

        doc_id = safe_console_text("".join(c for c in str(ch.get("doc_id") or "") if c.isprintable()))
        title = safe_console_text("".join(c for c in str(ch.get("title") or "") if c.isprintable()))
        filename = safe_console_text("".join(c for c in str(ch.get("filename") or "") if c.isprintable()))

        page_start = to_pos_int(ch.get("page_start"))
        page_end = to_pos_int(ch.get("page_end"))
        url = safe_console_text(build_source_url(ch, page_start))

        if not title:
            title = filename or (doc_id or "unknown-source")

        sources.append(
            {
                "n": i,
                "title": title,
                "filename": filename,
                "page_start": page_start,
                "page_end": page_end,
                "url": url,
            }
        )

    return {
        "answer": answer,
        "sources": sources,
        "retrieved": len(chunks),
        "chunks": chunks,
    }