"""собирает текстовый контекст для llm из найденных чанков:
добавляет к каждому чанку служебный заголовок с номером, doc_id, c
hunk_id, score и страницами, а затем объединяет все чанки в один общий контекст
для генерации ответа."""


from __future__ import annotations

from src.rag.common.text_utils import to_pos_int # pyright: ignore[reportMissingImports]


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, ch in enumerate(chunks, start=1):
        page_start = to_pos_int(ch.get("page_start"))
        page_end = to_pos_int(ch.get("page_end"))
        page_part = ""

        if page_start and page_end:
            page_part = f" page={page_start}" if page_start == page_end else f" pages={page_start}-{page_end}"

        header = f"[{i}] doc_id={ch['doc_id']} chunk={ch['chunk_id']} score={ch['score']:.4f}{page_part}"
        parts.append(header + "\n" + ch["text"])

    return "\n\n".join(parts)