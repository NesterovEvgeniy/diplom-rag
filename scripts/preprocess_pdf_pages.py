"""выполняет предобработку PDF-источников: постранично извлекает текст,
   очищает его от шума, номеров страниц и повторяющихся headers/footers,
   а затем сохраняет очищенный текст страниц в data/processed/page_texts/*.jsonl
   и формирует отчёты по результатам обработки.
"""


from __future__ import annotations

import csv
import json
import math
import re

from collections import Counter
from pathlib import Path
from typing import Iterable
from pypdf import PdfReader


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent

PDF_DIR = PROJECT_ROOT / "data" / "sources_raw" # папка с исходными PDF файлами
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "page_texts" # Папка, куда сохранять очищенный текст по страницам.
REPORT_DIR = PROJECT_ROOT / "reports" / "pdf_checks" # Папка для технических отчётов preprocessing

REPORT_CSV = REPORT_DIR / "pdf_preprocess_report.csv" # Итоговый CSV-отчёт по каждому PDF
REPORT_HEADERS_TXT = REPORT_DIR / "pdf_preprocess_headers_report.txt" # TXT-отчёт о найденных повторяющихся headers/footers

TOP_ZONE_LINES = 3 # Сколько верхних строк страницы считать зоной возможного header
BOTTOM_ZONE_LINES = 3 # Сколько нижних строк страницы считать зоной возможного footer

MIN_REPEAT_RATIO = 0.30 # Минимальная доля страниц, на которых строка должна повторяться, чтобы считаться header/footer. 30%
MIN_REPEATS = 3 # Минимальное абсолютное число повторов строки. Даже если 30% мало, меньше 3 повторов не считаем

MAX_HEADER_FOOTER_LEN = 180 # Максимальная длина строки-кандидата на header/footer. Слишком длинные строки не считаются колонтитулами


# делает имя файла безопасным для сохранения в файловой системе (слова через _ )
def make_safe_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]+', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

# очищает одну строку текста от мусорных символов и лишних пробелов
def normalize_line_spaces(line: str) -> str:
    line = line.replace("\x00", " ")
    line = line.replace("\xa0", " ")
    line = line.replace("\u200b", "")
    line = line.replace("\ufeff", "")
    line = re.sub(r"[ \t]+", " ", line)
    return line.strip()

# очищает весь сырой текст страницы целиком от мусорных символов и лишних пробелов
def normalize_text_preserve_newlines(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# склеивает слово, которое в PDF было разорвано переносом через дефис на новую строку
def fix_hyphen_linebreaks(text: str) -> str:

    return re.sub(r"([A-Za-zА-Яа-яЁё])-\n([A-Za-zА-Яа-яЁё])", r"\1\2", text)

# готовит строку для сравнения и подсчёта повторов headers/footers
def normalize_line_for_stats(line: str) -> str:
    line = normalize_line_spaces(line)
    line = line.lower()
    return line

# проверяет, похожа ли строка на номер страницы
def is_probable_page_number_line(line: str) -> bool:
    line = normalize_line_spaces(line)
    if not line:
        return False

    return bool(re.fullmatch(r"[\-\–\—\s]*\d{1,4}[\-\–\—\s]*", line))

# проверяет, является ли строка шумом, а не полезным текстом
def is_probable_noise_line(line: str) -> bool:
    line = normalize_line_spaces(line)
    if not line:
        return True

    return bool(re.fullmatch(r"[\-\–\—\_\.\,\:\;\*\·\•\s]+", line))

# безопасно извлекает текст из одной страницы PDF
def safe_extract_text(page) -> str:
    try:
        return page.extract_text() or ""
    except Exception:
        return ""

# превращает сырой текст страницы в список чистых полезных строк
def split_clean_lines(raw_text: str) -> list[str]:

    raw_text = fix_hyphen_linebreaks(raw_text)
    raw_text = normalize_text_preserve_newlines(raw_text)

    lines = raw_text.split("\n")
    cleaned_lines: list[str] = []

    for line in lines:
        line = normalize_line_spaces(line)

        if not line:
            continue
        if is_probable_noise_line(line):
            continue
        if is_probable_page_number_line(line):
            continue

        cleaned_lines.append(line)

    return cleaned_lines

# собирает очищенные строки для всех страниц PDF
def collect_page_lines(reader: PdfReader) -> list[list[str]]:
    all_pages_lines: list[list[str]] = []

    for page in reader.pages:
        raw_text = safe_extract_text(page)
        lines = split_clean_lines(raw_text)
        all_pages_lines.append(lines)

    return all_pages_lines

# автоматически ищет повторяющиеся колонтитулы: что считать header, а что footer
def detect_repeated_headers_footers(
    pages_lines: list[list[str]],
    top_zone_lines: int = TOP_ZONE_LINES,
    bottom_zone_lines: int = BOTTOM_ZONE_LINES,
    min_ratio: float = MIN_REPEAT_RATIO,
    min_repeats: int = MIN_REPEATS,
    max_line_len: int = MAX_HEADER_FOOTER_LEN,
) -> tuple[set[str], set[str]]:

    total_pages = len(pages_lines)
    threshold = max(min_repeats, math.ceil(total_pages * min_ratio))

    top_counter: Counter[str] = Counter()
    bottom_counter: Counter[str] = Counter()

    for lines in pages_lines:
        if not lines:
            continue

        top_candidates = lines[:top_zone_lines]
        bottom_candidates = lines[-bottom_zone_lines:]

        top_unique = {
            normalize_line_for_stats(line)
            for line in top_candidates
            if line and len(normalize_line_spaces(line)) <= max_line_len
        }
        bottom_unique = {
            normalize_line_for_stats(line)
            for line in bottom_candidates
            if line and len(normalize_line_spaces(line)) <= max_line_len
        }

        for line in top_unique:
            top_counter[line] += 1

        for line in bottom_unique:
            bottom_counter[line] += 1

    repeated_headers: set[str] = set()
    repeated_footers: set[str] = set()

    all_candidates = set(top_counter.keys()) | set(bottom_counter.keys())

    for line in all_candidates:
        if len(line) < 3:
            continue

        top_count = top_counter.get(line, 0)
        bottom_count = bottom_counter.get(line, 0)

        if top_count >= threshold and top_count > bottom_count:
            repeated_headers.add(line)

        if bottom_count >= threshold and bottom_count > top_count:
            repeated_footers.add(line)

    return repeated_headers, repeated_footers

# удаляет найденные headers/footers со страницы и делает финальную очистку строк
def clean_page_lines(
    lines: list[str],
    repeated_headers: set[str],
    repeated_footers: set[str],
    top_zone_lines: int = TOP_ZONE_LINES,
    bottom_zone_lines: int = BOTTOM_ZONE_LINES,
) -> list[str]:

    cleaned = list(lines)

    top_limit = min(top_zone_lines, len(cleaned))
    for i in range(top_limit):
        key = normalize_line_for_stats(cleaned[i])
        if key in repeated_headers:
            cleaned[i] = ""

    start_bottom = max(0, len(cleaned) - bottom_zone_lines)
    for i in range(start_bottom, len(cleaned)):
        key = normalize_line_for_stats(cleaned[i])
        if key in repeated_footers:
            cleaned[i] = ""

    final_lines: list[str] = []

    for line in cleaned:
        line = normalize_line_spaces(line)
        if not line:
            continue

        line = re.sub(r"\s+([,.;:!?])", r"\1", line)
        line = re.sub(r"([(\[\{])\s+", r"\1", line)
        line = re.sub(r"\s+([)\]\}])", r"\1", line)

        if not line:
            continue

        final_lines.append(line)

    return final_lines

# собирает список строк обратно в единый текст страницы
def join_lines_to_text(lines: Iterable[str]) -> str:
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# функция preprocessing одного PDF-файла (берёт один PDF и делает из него page_texts.jsonl + собирает статистику по обработке preprocess_pdf_pages)
def preprocess_pdf(pdf_path: Path) -> tuple[dict, dict]:
    result = {
        "filename": pdf_path.name,
        "output_file": "",
        "total_pages": 0,
        "raw_nonempty_pages": 0,
        "clean_nonempty_pages": 0,
        "empty_after_cleaning_pages": 0,
        "repeated_headers_detected": 0,
        "repeated_footers_detected": 0,
        "status": "ok",
        "error": "",
    }

    detected_info = {
        "headers": set(),
        "footers": set(),
    }

    try:
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
        result["total_pages"] = total_pages

        pages_lines = collect_page_lines(reader)
        repeated_headers, repeated_footers = detect_repeated_headers_footers(pages_lines)

        detected_info["headers"] = repeated_headers
        detected_info["footers"] = repeated_footers

        result["repeated_headers_detected"] = len(repeated_headers)
        result["repeated_footers_detected"] = len(repeated_footers)

        safe_stem = make_safe_filename(pdf_path.stem)
        output_path = OUTPUT_DIR / f"{safe_stem}.jsonl"
        result["output_file"] = str(output_path.relative_to(PROJECT_ROOT))

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        raw_nonempty_pages = 0
        clean_nonempty_pages = 0
        empty_after_cleaning_pages = 0

        with output_path.open("w", encoding="utf-8") as f:
            for page_num, lines in enumerate(pages_lines, start=1):
                raw_text = join_lines_to_text(lines)
                if raw_text:
                    raw_nonempty_pages += 1

                clean_lines = clean_page_lines(
                    lines=lines,
                    repeated_headers=repeated_headers,
                    repeated_footers=repeated_footers,
                )
                cleaned_text = join_lines_to_text(clean_lines)

                if cleaned_text:
                    clean_nonempty_pages += 1
                else:
                    empty_after_cleaning_pages += 1

                record = {
                    "doc_id": safe_stem,
                    "source_pdf": pdf_path.name,
                    "title": pdf_path.stem,
                    "page_num": page_num,
                    "text": cleaned_text,
                    "raw_chars": len(raw_text),
                    "cleaned_chars": len(cleaned_text),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        result["raw_nonempty_pages"] = raw_nonempty_pages
        result["clean_nonempty_pages"] = clean_nonempty_pages
        result["empty_after_cleaning_pages"] = empty_after_cleaning_pages

        return result, detected_info

    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result, detected_info

# сохраняет общий CSV-отчёт по обработанным PDF
def save_csv(results: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "filename",
        "output_file",
        "total_pages",
        "raw_nonempty_pages",
        "clean_nonempty_pages",
        "empty_after_cleaning_pages",
        "repeated_headers_detected",
        "repeated_footers_detected",
        "status",
        "error",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

# сохраняет подробный текстовый отчёт о найденных headers и footers по каждому PDF
def save_headers_report(headers_map: dict[str, dict], txt_path: Path) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    for filename, payload in headers_map.items():
        headers = sorted(payload["headers"])
        footers = sorted(payload["footers"])

        lines.append("=" * 120)
        lines.append(f"Файл: {filename}")
        lines.append(f"Headers найдено: {len(headers)}")
        if headers:
            for item in headers:
                lines.append(f"[HEADER] {item}")
        else:
            lines.append("[HEADER] нет")

        lines.append(f"Footers найдено: {len(footers)}")
        if footers:
            for item in footers:
                lines.append(f"[FOOTER] {item}")
        else:
            lines.append("[FOOTER] нет")

        lines.append("")

    txt_path.write_text("\n".join(lines), encoding="utf-8")

# печатает краткую сводку preprocessing в консоль
def print_summary(results: list[dict]) -> None:
    print("\n=== PREPROCESS PDF -> CLEANED PAGE TEXTS V2 ===\n")
    header = (
        f"{'Статус':<10}"
        f"{'Страниц':<10}"
        f"{'Raw>0':<10}"
        f"{'Clean>0':<10}"
        f"{'EmptyAfter':<12}"
        f"{'Headers':<10}"
        f"{'Footers':<10}"
        f"Файл"
    )
    print(header)
    print("-" * len(header))

    for item in results:
        print(
            f"{item['status']:<10}"
            f"{item['total_pages']:<10}"
            f"{item['raw_nonempty_pages']:<10}"
            f"{item['clean_nonempty_pages']:<10}"
            f"{item['empty_after_cleaning_pages']:<12}"
            f"{item['repeated_headers_detected']:<10}"
            f"{item['repeated_footers_detected']:<10}"
            f"{item['filename']}"
        )

# печать в консоль короткую итоговую таблицу по всем PDF
def main() -> None:
    if not PDF_DIR.exists():
        print(f"Папка не найдена: {PDF_DIR}")
        return

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"В папке нет PDF-файлов: {PDF_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Корень проекта: {PROJECT_ROOT}")
    print(f"Папка с PDF: {PDF_DIR}")
    print(f"Папка с результатом: {OUTPUT_DIR}")
    print(f"Папка с отчетами: {REPORT_DIR}")
    print(f"Найдено PDF-файлов: {len(pdf_files)}")

    results: list[dict] = []
    headers_map: dict[str, dict] = {}

    for pdf_path in pdf_files:
        result, detected_info = preprocess_pdf(pdf_path)
        results.append(result)
        headers_map[pdf_path.name] = detected_info

    print_summary(results)
    save_csv(results, REPORT_CSV)
    save_headers_report(headers_map, REPORT_HEADERS_TXT)

    print(f"\nCSV-отчет сохранен: {REPORT_CSV}")
    print(f"TXT-отчет по headers/footers сохранен: {REPORT_HEADERS_TXT}")


if __name__ == "__main__":
    main()