"""проверяет качество PDF-источников перед обработкой:
считает количество страниц с текстом и пустых страниц,
определяет общий статус PDF,
выводит краткий отчет и сохраняет результаты в csv."""


from __future__ import annotations

import csv
import re

from pathlib import Path
from pypdf import PdfReader


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent
PDF_DIR = PROJECT_ROOT / "data" / "sources_raw"
REPORT_DIR = PROJECT_ROOT / "reports" / "pdf_checks"
REPORT_CSV = REPORT_DIR / "pdf_check_report.csv"


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_preview(text: str, max_len: int = 200) -> str:
    text = normalize_text(text)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def decide_status(
    total_pages: int,
    text_pages: int,
    empty_pages: int,
    first_preview: str,
) -> str:
    if total_pages == 0 or text_pages == 0:
        return "bad"

    empty_ratio = empty_pages / total_pages if total_pages else 1.0

    if not first_preview or len(first_preview) < 30:
        return "acceptable"

    if empty_ratio > 0.5:
        return "bad"

    if empty_ratio > 0.2:
        return "acceptable"

    return "good"


def analyze_pdf(pdf_path: Path) -> dict:
    result = {
        "filename": pdf_path.name,
        "status": "bad",
        "total_pages": 0,
        "text_pages": 0,
        "empty_pages": 0,
        "first_text_page": "",
        "preview": "",
        "error": "",
    }

    try:
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)

        text_pages = 0
        empty_pages = 0
        first_text_page = ""
        preview = ""

        for page_index, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            cleaned = normalize_text(raw_text)

            if cleaned:
                text_pages += 1
                if not first_text_page:
                    first_text_page = str(page_index)
                    preview = make_preview(cleaned)
            else:
                empty_pages += 1

        status = decide_status(
            total_pages=total_pages,
            text_pages=text_pages,
            empty_pages=empty_pages,
            first_preview=preview,
        )

        result.update(
            {
                "status": status,
                "total_pages": total_pages,
                "text_pages": text_pages,
                "empty_pages": empty_pages,
                "first_text_page": first_text_page,
                "preview": preview,
            }
        )
        return result

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def print_summary(results: list[dict]) -> None:
    print("\n=== КРАТКИЙ ОТЧЕТ ПО PDF ===\n")
    header = (
        f"{'Статус':<12}"
        f"{'Страниц':<10}"
        f"{'С текстом':<12}"
        f"{'Пустых':<10}"
        f"{'Первая':<10}"
        f"Файл"
    )
    print(header)
    print("-" * len(header))

    for item in results:
        print(
            f"{item['status']:<12}"
            f"{item['total_pages']:<10}"
            f"{item['text_pages']:<12}"
            f"{item['empty_pages']:<10}"
            f"{str(item['first_text_page']):<10}"
            f"{item['filename']}"
        )

    print("\n=== ДЕТАЛИ ===\n")

    for item in results:
        print(f"Файл: {item['filename']}")
        print(f"Статус: {item['status']}")
        print(f"Страниц всего: {item['total_pages']}")
        print(f"Страниц с текстом: {item['text_pages']}")
        print(f"Пустых страниц: {item['empty_pages']}")
        print(f"Первая непустая страница: {item['first_text_page'] or '-'}")

        if item["error"]:
            print(f"Ошибка: {item['error']}")
        else:
            print(f"Фрагмент текста: {item['preview'] or '-'}")

        print("-" * 100)


def save_csv(results: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "filename",
        "status",
        "total_pages",
        "text_pages",
        "empty_pages",
        "first_text_page",
        "preview",
        "error",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    if not PDF_DIR.exists():
        print(f"Папка не найдена: {PDF_DIR}")
        return

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"В папке нет PDF-файлов: {PDF_DIR}")
        return

    print(f"Корень проекта: {PROJECT_ROOT}")
    print(f"Папка с PDF: {PDF_DIR}")
    print(f"Папка для отчета: {REPORT_DIR}")
    print(f"Найдено PDF-файлов: {len(pdf_files)}")

    results = [analyze_pdf(pdf_path) for pdf_path in pdf_files]

    print_summary(results)
    save_csv(results, REPORT_CSV)

    print(f"\nCSV-отчет сохранен: {REPORT_CSV}")


if __name__ == "__main__":
    main()