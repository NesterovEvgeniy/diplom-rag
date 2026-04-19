"""выполняет чанкирование очищенных постраничных текстов:
читает page_texts из data/processed/page_texts,
разбивает текст каждой страницы на локальные чанки,
сохраняет результат в data/processed/chunks/chunks.jsonl
и формирует статистику по чанкам."""


from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


INPUT_DIR = Path("data/processed/page_texts")
OUTPUT_DIR = Path("data/processed/chunks")
OUTPUT_JSONL = OUTPUT_DIR / "chunks.jsonl"
OUTPUT_STATS = OUTPUT_DIR / "chunk_stats.json"


@dataclass
class PageRecord:
    doc_id: str
    title: str
    source_pdf: str
    page_num: int
    text: str


def _slugify_filename_stem(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^\w\-]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown_doc"


def _read_json_file(path: Path) -> list[Any]:
    if path.suffix.lower() == ".jsonl":
        items: list[Any] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception as e:
                    raise ValueError(f"{path}: invalid JSONL at line {line_no}: {e}") from e
        return items

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return [data]

    return []


def _pick_str(obj: dict[str, Any], keys: list[str], default: str = "") -> str:
    for key in keys:
        value = obj.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def _pick_int(obj: dict[str, Any], keys: list[str]) -> int | None:
    for key in keys:
        value = obj.get(key)
        if value is None:
            continue
        try:
            number = int(value)
        except Exception:
            continue
        if number > 0:
            return number
    return None


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    paragraphs: list[str] = []
    for raw_para in re.split(r"\n\s*\n+", text):
        para = re.sub(r"\s+", " ", raw_para).strip()
        if para:
            paragraphs.append(para)

    return "\n\n".join(paragraphs).strip()


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    parts = re.split(r'(?<=[.!?…])\s+(?=[A-ZА-ЯЁ0-9"«(])', text)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


def _split_by_words(text: str, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current = ""

    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = word

            while len(current) > max_chars:
                chunks.append(current[:max_chars].strip())
                current = current[max_chars:].strip()

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if c]


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    sentence = re.sub(r"\s+", " ", sentence).strip()
    if not sentence:
        return []

    if len(sentence) <= max_chars:
        return [sentence]

    soft_parts = re.split(r"(?<=[;:])\s+|(?<=,)\s+", sentence)
    soft_parts = [p.strip() for p in soft_parts if p.strip()]

    if len(soft_parts) > 1:
        chunks: list[str] = []
        current = ""

        for part in soft_parts:
            candidate = part if not current else f"{current} {part}"
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())

                if len(part) <= max_chars:
                    current = part
                else:
                    word_chunks = _split_by_words(part, max_chars)
                    if word_chunks:
                        chunks.extend(word_chunks[:-1])
                        current = word_chunks[-1]
                    else:
                        current = ""

        if current:
            chunks.append(current.strip())

        if chunks and all(len(ch) <= max_chars for ch in chunks):
            return [ch for ch in chunks if ch]

    return _split_by_words(sentence, max_chars)


def chunk_page_text(
    text: str,
    chunk_size: int = 900,
    overlap: int = 120,
    min_chunk_chars: int = 180,
) -> list[str]:
    del overlap

    text = _normalize_text(text)
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        sentences = _split_sentences(para)
        if not sentences:
            continue

        for sent in sentences:
            sent_parts = _split_long_sentence(sent, chunk_size)

            for sent_part in sent_parts:
                if not current:
                    current = sent_part
                    continue

                candidate = f"{current} {sent_part}".strip()

                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    chunks.append(current.strip())
                    current = sent_part

        if current and len(current) >= int(chunk_size * 0.6):
            chunks.append(current.strip())
            current = ""

    if current:
        chunks.append(current.strip())

    if len(chunks) >= 2 and len(chunks[-1]) < min_chunk_chars:
        merged = f"{chunks[-2]} {chunks[-1]}".strip()
        if len(merged) <= int(chunk_size * 1.35):
            chunks[-2] = merged
            chunks.pop()

    return [ch for ch in chunks if ch.strip()]


def _extract_page_records_from_obj(
    obj: Any,
    fallback_doc_id: str,
    fallback_title: str,
    fallback_source_pdf: str,
) -> list[PageRecord]:
    if not isinstance(obj, dict):
        return []

    for pages_key in ("pages", "items", "records"):
        pages_value = obj.get(pages_key)
        if isinstance(pages_value, list):
            parent_doc_id = _pick_str(obj, ["doc_id"], fallback_doc_id)
            parent_title = _pick_str(obj, ["title"], fallback_title)
            parent_source_pdf = _pick_str(
                obj,
                ["source_pdf", "filename", "pdf_file", "pdf_name", "file", "source"],
                fallback_source_pdf,
            )

            out: list[PageRecord] = []
            for page_obj in pages_value:
                out.extend(
                    _extract_page_records_from_obj(
                        page_obj,
                        fallback_doc_id=parent_doc_id,
                        fallback_title=parent_title,
                        fallback_source_pdf=parent_source_pdf,
                    )
                )
            return out

    doc_id = _pick_str(obj, ["doc_id"], fallback_doc_id)
    title = _pick_str(obj, ["title"], fallback_title)
    source_pdf = _pick_str(
        obj,
        ["source_pdf", "filename", "pdf_file", "pdf_name", "file", "source"],
        fallback_source_pdf,
    )
    page_num = _pick_int(obj, ["page_num", "page", "page_no", "page_number", "pdf_page"])
    text = _pick_str(obj, ["text", "cleaned_text", "page_text", "content"])

    if not doc_id:
        doc_id = fallback_doc_id
    if not title:
        title = source_pdf or fallback_title or doc_id
    if not source_pdf:
        source_pdf = fallback_source_pdf or title

    if page_num is None or not text.strip():
        return []

    return [
        PageRecord(
            doc_id=doc_id,
            title=title,
            source_pdf=source_pdf,
            page_num=page_num,
            text=text,
        )
    ]


def read_page_records(input_dir: Path) -> list[PageRecord]:
    records: list[PageRecord] = []

    files = sorted(
        [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}]
    )
    if not files:
        raise FileNotFoundError(f"No .json/.jsonl files found in {input_dir}")

    for path in files:
        fallback_source_pdf = path.stem
        fallback_doc_id = _slugify_filename_stem(path.stem)
        fallback_title = path.stem

        items = _read_json_file(path)
        for obj in items:
            records.extend(
                _extract_page_records_from_obj(
                    obj,
                    fallback_doc_id=fallback_doc_id,
                    fallback_title=fallback_title,
                    fallback_source_pdf=fallback_source_pdf,
                )
            )

    records.sort(key=lambda r: (r.doc_id, r.page_num))
    return records


def build_chunk_rows(
    page_records: Iterable[PageRecord],
    chunk_size: int,
    overlap: int,
    min_chunk_chars: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_doc_page_counter: dict[tuple[str, int], int] = {}

    for rec in page_records:
        page_chunks = chunk_page_text(
            rec.text,
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_chars=min_chunk_chars,
        )
        if not page_chunks:
            continue

        for chunk_text_value in page_chunks:
            key = (rec.doc_id, rec.page_num)
            per_doc_page_counter[key] = per_doc_page_counter.get(key, 0) + 1
            chunk_idx = per_doc_page_counter[key]

            chunk_id = f"{rec.doc_id}_p{rec.page_num:04d}_c{chunk_idx:03d}"

            rows.append(
                {
                    "doc_id": rec.doc_id,
                    "title": rec.title,
                    "source_pdf": rec.source_pdf,
                    "page_num": rec.page_num,
                    "page_start": rec.page_num,
                    "page_end": rec.page_num,
                    "chunk_index": chunk_idx,
                    "chunk_id": chunk_id,
                    "text": chunk_text_value,
                    "chars": len(chunk_text_value),
                }
            )

    return rows


def save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_stats(page_records: list[PageRecord], chunk_rows: list[dict[str, Any]]) -> dict[str, Any]:
    docs: dict[str, dict[str, Any]] = {}

    for rec in page_records:
        doc = docs.setdefault(
            rec.doc_id,
            {
                "title": rec.title,
                "source_pdf": rec.source_pdf,
                "pages_with_text": 0,
                "chunks": 0,
            },
        )
        doc["pages_with_text"] += 1

    for row in chunk_rows:
        doc = docs.setdefault(
            row["doc_id"],
            {
                "title": row["title"],
                "source_pdf": row["source_pdf"],
                "pages_with_text": 0,
                "chunks": 0,
            },
        )
        doc["chunks"] += 1

    char_lengths = [int(row["chars"]) for row in chunk_rows]
    stats = {
        "total_docs": len(docs),
        "total_pages_with_text": len(page_records),
        "total_chunks": len(chunk_rows),
        "chunk_chars_min": min(char_lengths) if char_lengths else 0,
        "chunk_chars_max": max(char_lengths) if char_lengths else 0,
        "chunk_chars_avg": round(sum(char_lengths) / len(char_lengths), 2) if char_lengths else 0,
        "docs": docs,
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build page-local chunks from processed page_texts.")
    parser.add_argument(
        "--input-dir",
        default=str(INPUT_DIR),
        help="Directory with processed page_texts JSON/JSONL",
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(OUTPUT_JSONL),
        help="Output chunks.jsonl path",
    )
    parser.add_argument(
        "--output-stats",
        default=str(OUTPUT_STATS),
        help="Output chunk_stats.json path",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=900,
        help="Target chunk size in characters",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=120,
        help="Compatibility parameter; not actively used in sentence-aware mode",
    )
    parser.add_argument(
        "--min-chunk-chars",
        type=int,
        default=180,
        help="Merge very small tail chunk into previous",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_jsonl = Path(args.output_jsonl)
    output_stats = Path(args.output_stats)

    page_records = read_page_records(input_dir)
    chunk_rows = build_chunk_rows(
        page_records=page_records,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        min_chunk_chars=args.min_chunk_chars,
    )

    save_jsonl(chunk_rows, output_jsonl)

    stats = build_stats(page_records, chunk_rows)
    output_stats.parent.mkdir(parents=True, exist_ok=True)
    output_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Input pages with text: {stats['total_pages_with_text']}")
    print(f"Output chunks: {stats['total_chunks']}")
    print(f"Chunks JSONL: {output_jsonl}")
    print(f"Stats JSON: {output_stats}")


if __name__ == "__main__":
    main()