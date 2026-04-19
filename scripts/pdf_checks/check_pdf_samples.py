"""проверяет качество PDF-источников перед обработкой:
считает количество страниц с текстом и пустых страниц,
определяет общий статус PDF,
выводит краткий отчет и сохраняет результаты в csv."""


from __future__ import annotations

import re

from pathlib import Path
from pypdf import PdfReader


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
PDF_DIR = PROJECT_ROOT / "data" / "sources_raw"
REPORT_DIR = PROJECT_ROOT / "reports" / "pdf_checks"
REPORT_TXT = REPORT_DIR / "pdf_page_samples_report.txt"


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_preview(text: str, max_len: int = 400) -> str:
    text = normalize_text(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def safe_extract(page) -> str:
    try:
        return normalize_text(page.extract_text() or "")
    except Exception:
        return ""


def build_range(start: int, end: int, total_pages: int) -> list[int]:
    start = max(0, start)
    end = min(total_pages, end)
    if start >= end:
        return []
    return list(range(start, end))


def find_first_non_empty(reader: PdfReader, page_indices: list[int]) -> tuple[int | None, str]:
    for idx in page_indices:
        text = safe_extract(reader.pages[idx])
        if text:
            return idx + 1, make_preview(text)
    return None, ""


def analyze_pdf(pdf_path: Path) -> str:
    lines: list[str] = []
    lines.append("=" * 120)
    lines.append(f"Файл: {pdf_path.name}")

    try:
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
        lines.append(f"Всего страниц: {total_pages}")

        begin_range = build_range(0, min(10, total_pages), total_pages)
        middle = total_pages // 2
        mid_range = build_range(middle - 5, middle + 5, total_pages)
        end_range = build_range(max(0, total_pages - 10), total_pages, total_pages)

        begin_page, begin_text = find_first_non_empty(reader, begin_range)
        mid_page, mid_text = find_first_non_empty(reader, mid_range)
        end_page, end_text = find_first_non_empty(reader, end_range)

        lines.append("")
        lines.append("[НАЧАЛО]")
        lines.append(f"Страница: {begin_page if begin_page is not None else '-'}")
        lines.append(f"Фрагмент: {begin_text or '-'}")

        lines.append("")
        lines.append("[СЕРЕДИНА]")
        lines.append(f"Страница: {mid_page if mid_page is not None else '-'}")
        lines.append(f"Фрагмент: {mid_text or '-'}")

        lines.append("")
        lines.append("[КОНЕЦ]")
        lines.append(f"Страница: {end_page if end_page is not None else '-'}")
        lines.append(f"Фрагмент: {end_text or '-'}")

        lines.append("")
        lines.append("[ПРЕДВАРИТЕЛЬНАЯ ОЦЕНКА]")
        if begin_text and mid_text and end_text:
            lines.append("Статус: текст найден в начале, середине и конце документа.")
        elif begin_text and mid_text:
            lines.append("Статус: текст найден в начале и середине, конец нужно проверить вручную.")
        elif mid_text or end_text:
            lines.append("Статус: текст извлекается нестабильно, нужна ручная проверка.")
        else:
            lines.append("Статус: текст извлекается плохо, нужна дополнительная диагностика.")

    except Exception as exc:
        lines.append(f"Ошибка: {type(exc).__name__}: {exc}")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    if not PDF_DIR.exists():
        print(f"Папка не найдена: {PDF_DIR}")
        return

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"В папке нет PDF-файлов: {PDF_DIR}")
        return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Корень проекта: {PROJECT_ROOT}")
    print(f"Папка с PDF: {PDF_DIR}")
    print(f"Папка для отчета: {REPORT_DIR}")
    print(f"Найдено PDF-файлов: {len(pdf_files)}")
    print()

    report_blocks: list[str] = []

    for pdf_path in pdf_files:
        block = analyze_pdf(pdf_path)
        report_blocks.append(block)
        print(block)

    report_text = "\n".join(report_blocks)
    REPORT_TXT.write_text(report_text, encoding="utf-8")

    print("=" * 120)
    print(f"TXT-отчет сохранен: {REPORT_TXT}")


if __name__ == "__main__":
    main()