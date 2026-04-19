"""выполняет постобработку baseline-run результатов:
читает jsonl-файл с результатами оценки,
собирает удобную csv-таблицу для ручного анализа ошибок
и считает краткую сводку по попаданиям в источник и страницу."""


from __future__ import annotations

from datetime import datetime

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def find_repo_root(start_from: Path) -> Path:
    start = start_from.resolve()
    for candidate in [start] + list(start.parents):
        if (candidate / "data").exists() and (candidate / "scripts").exists():
            return candidate
    return start.parent if start.is_file() else start


def resolve_path(path_str: str | None, repo_root: Path) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str)
    return path if path.is_absolute() else repo_root / path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Ошибка JSONL в строке {line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Строка {line_no}: ожидался JSON-объект.")
            rows.append(obj)
    return rows


def normalize_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def stringify_list(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    return ",".join(str(x) for x in value)

def safe_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for x in value:
        try:
            result.append(int(x))
        except (TypeError, ValueError):
            continue
    return result


def bool_to_str(value: Any) -> str:
    if value is True:
        return "True"
    if value is False:
        return "False"
    return ""

def extract_alt_ref_pages(record: dict[str, Any]) -> list[int]:
    gold_references = record.get("gold_references")
    if not isinstance(gold_references, list):
        return []

    pages: set[int] = set()

    for ref in gold_references:
        if not isinstance(ref, dict):
            continue
        if str(ref.get("ref_kind", "")).strip() != "alternative":
            continue

        for page in safe_int_list(ref.get("source_pages")):
            pages.add(page)

    return sorted(pages)


def shorten(text: str, limit: int = 220) -> str:
    text = normalize_text(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def find_latest_run_jsonl(runs_dir: Path) -> Path:
    candidates = sorted(
        [
            p
            for p in runs_dir.glob("baseline*.jsonl")
            if ".analysis." not in p.name
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"Не найдено baseline *.jsonl файлов в папке: {runs_dir}"
        )
    return candidates[0]


def build_analysis_rows(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for record in records:
        row = {
            "question_id": str(record.get("question_id", "")).strip(),
            "question": normalize_text(str(record.get("question", "")).strip()),
            "gold_source_filename": normalize_text(str(record.get("source_filename", "")).strip()),
            "gold_pages_prim_ref": stringify_list(record.get("gold_source_pages")),
            "gold_pages_alt_ref": stringify_list(extract_alt_ref_pages(record)),
            "gold_source_match_found": bool_to_str(
                record.get("gold_source_match_found_any", record.get("gold_source_match_found"))
            ),
            "gold_source_cited_pages": stringify_list(
                record.get("cited_source_pages_any_gold", record.get("cited_source_pages"))
            ),
            "page_hit": bool_to_str(
                record.get("page_hit_any", record.get("page_hit"))
            ),
            "model_answer": normalize_text(str(record.get("model_answer", "")).strip()),
            "manual_error_type": "",
        }
        rows.append(row)

    return rows


def save_csv(rows: list[dict[str, str]], path: Path) -> None:
    fieldnames = [
    "question_id",
    "question",
    "gold_source_filename",
    "gold_pages_prim_ref",
    "gold_pages_alt_ref",
    "gold_source_match_found",
    "gold_source_cited_pages",
    "page_hit",
    "model_answer",
    "manual_error_type",
]

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def save_md(rows: list[dict[str, str]], path: Path) -> None:
    headers = [
        "question_id",
        "question",
        "gold_source_filename",
        "gold_pages_prim_ref",
        "gold_pages_alt_ref",
        "gold_source_match_found",
        "gold_source_cited_pages",
        "page_hit",
        "model_answer",
        "manual_error_type",
    ]

    def esc(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", "<br>")

    with path.open("w", encoding="utf-8") as f:
        f.write("# Baseline error analysis\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")

        for row in rows:
            values = [
                esc(row["question_id"]),
                esc(row["question"]),
                esc(row["gold_source_filename"]),
                esc(row["gold_pages_prim_ref"]),
                esc(row["gold_pages_alt_ref"]),
                esc(row["gold_source_match_found"]),
                esc(row["gold_source_cited_pages"]),
                esc(row["page_hit"]),
                esc(shorten(row["model_answer"], 180)),
                esc(row["manual_error_type"]),
            ]
            f.write("| " + " | ".join(values) + " |\n")


def save_analysis_jsonl(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Собирает таблицу для ошибкоразбора baseline-run."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Путь до baseline-run JSONL. Если не указан, будет взят последний baseline*.jsonl из data/evaluation/runs.",
    )
    parser.add_argument(
        "--runs-dir",
        default="data/evaluation/runs",
        help="Папка с baseline-run файлами.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Префикс выходных файлов без расширения. По умолчанию создаётся рядом с input.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Корень репозитория. Если не указан, ищется автоматически.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_path = Path(__file__).resolve()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root(script_path)

    runs_dir = resolve_path(args.runs_dir, repo_root)
    if runs_dir is None:
        raise ValueError("Не удалось определить runs_dir.")

    input_path = resolve_path(args.input, repo_root)
    if input_path is None:
        input_path = find_latest_run_jsonl(runs_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Не найден input файл: {input_path}")

    records = load_jsonl(input_path)
    rows = build_analysis_rows(records)

    if args.output_prefix:
        output_prefix = resolve_path(args.output_prefix, repo_root)
        if output_prefix is None:
            raise ValueError("Не удалось определить output_prefix.")
    else:
        suffix = input_path.suffix
        stem = input_path.name[: -len(suffix)] if suffix else input_path.name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = input_path.with_name(f"{stem}_{timestamp}")

    csv_path = output_prefix.with_suffix(".csv")

    save_csv(rows, csv_path)

    total = len(rows)
    page_hit_true = sum(row["page_hit"] == "True" for row in rows)
    page_hit_false = sum(row["page_hit"] == "False" for row in rows)
    gold_source_match_true = sum(row["gold_source_match_found"] == "True" for row in rows)

    print(f"[analysis] input                  = {input_path}")
    print(f"[analysis] csv                    = {csv_path}")
    print(f"[analysis] total                  = {total}")
    print(f"[analysis] gold_source_match_true = {gold_source_match_true}")
    print(f"[analysis] page_hit_true          = {page_hit_true}")
    print(f"[analysis] page_hit_false         = {page_hit_false}")


if __name__ == "__main__":
    main()