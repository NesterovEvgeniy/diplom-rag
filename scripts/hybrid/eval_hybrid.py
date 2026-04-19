"""выполняет автоматическую оценку hybrid rag на наборе gold-вопросов:
запускает ask_hybrid для каждого вопроса, извлекает ответ и источники из stdout,
сравнивает результат с gold-разметкой,
сохраняет подробный run в jsonl и итоговую summary-статистику."""


from __future__ import annotations

import unicodedata
import argparse
import json
import os
import re
import subprocess
import sys
import time

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
SOURCE_START_RE = re.compile(r"^\s*(?:\[(\d+)\]|(\d+)[\].)])\s*(.*)$")
ANSWER_MARKER_RE = re.compile(r"^\s*(?:answer|ответ)\s*:\s*(.*)$", re.IGNORECASE)
SOURCES_MARKER_RE = re.compile(
    r"^\s*(?:sources|источники|evidence|доказательства)\s*:\s*$",
    re.IGNORECASE,
)
CITATION_RE = re.compile(r"\[(\d+)\]")
PDF_NAME_RE = re.compile(r'([^\s"<>\'|]+\.pdf)\b', re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)

REFUSAL_NORMALIZED = "в источниках нет ответа"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text or "")


def normalize_spaces(text: str) -> str:
    text = strip_ansi(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_compare(text: str) -> str:
    text = normalize_spaces(text).lower().strip()
    text = text.strip(" .!?:;\"'«»()[]{}")
    return text


def is_refusal_answer(answer: str) -> bool:
    return normalize_for_compare(answer) == REFUSAL_NORMALIZED


def resolve_path(path_str: str, repo_root: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else repo_root / path


def find_repo_root(start_from: Path) -> Path:
    start = start_from.resolve()
    candidates = [start] + list(start.parents)
    for candidate in candidates:
        if (candidate / "src").exists() and (candidate / "data").exists():
            return candidate
    return start.parent if start.is_file() else start


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Ошибка JSONL в строке {line_no}: {e}") from e
            if not isinstance(item, dict):
                raise ValueError(f"Строка {line_no}: ожидается JSON-объект.")
            items.append(item)
    return items


def parse_answer(stdout_text: str) -> str:
    text = normalize_spaces(stdout_text)
    if not text:
        return ""

    refusal_patterns = [
        r"в источниках нет ответа(?: на этот вопрос)?\.?",
    ]

    for pattern in refusal_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return "В источниках нет ответа."

    lines = text.split("\n")

    answer_started = False
    answer_lines: list[str] = []

    for line in lines:
        match = ANSWER_MARKER_RE.match(line)
        if match:
            answer_started = True
            inline = match.group(1).strip()
            if inline:
                answer_lines.append(inline)
            continue

        if answer_started:
            if SOURCES_MARKER_RE.match(line):
                break
            answer_lines.append(line)

    if answer_started:
        return normalize_spaces("\n".join(answer_lines))

    fallback_lines: list[str] = []
    for line in lines:
        if SOURCES_MARKER_RE.match(line):
            break
        fallback_lines.append(line)

    fallback = normalize_spaces("\n".join(fallback_lines))

    if re.search(r"в источниках нет ответа", fallback, flags=re.IGNORECASE):
        return "В источниках нет ответа."

    return fallback


def parse_page_token(token: str) -> list[int]:
    token = token.strip().replace("–", "-").replace("—", "-")
    if not token:
        return []

    if "-" in token:
        parts = token.split("-", 1)
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            start = int(parts[0])
            end = int(parts[1])
            if start <= end and (end - start) <= 20:
                return list(range(start, end + 1))
            return [start, end]
        return []

    if token.isdigit():
        return [int(token)]

    return []

def _extract_structured_field(raw: str, field: str, next_field: str | None = None) -> str | None:
    if next_field is not None:
        pattern = rf"{re.escape(field)}=(.*?)\s+\|\s+{re.escape(next_field)}="
    else:
        pattern = rf"{re.escape(field)}=(.*)$"

    m = re.search(pattern, raw, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None

    value = m.group(1).strip()
    return value or None


def extract_page_numbers(text: str) -> list[int]:
    pages: set[int] = set()
    clean = strip_ansi(text or "")

    for match in re.finditer(r"#page=(\d+)", clean, flags=re.IGNORECASE):
        pages.add(int(match.group(1)))

    for match in re.finditer(
        r"(?:стр(?:аница|\.|)|page)\s*[:=]?\s*([\d,\s\-–—]+)",
        clean,
        flags=re.IGNORECASE,
    ):
        expr = match.group(1)
        parts = [p for p in re.split(r"[,\s]+", expr) if p]
        for part in parts:
            for page in parse_page_token(part):
                pages.add(page)

    return sorted(pages)


def normalize_filename(value: str | None) -> str:
    if not value:
        return ""

    value = unquote(str(value)).strip().strip("\"' ")
    value = value.replace("\\", "/")
    value = value.split("#", 1)[0].split("?", 1)[0]
    value = Path(value).name

    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = re.sub(r"\s+", " ", value).strip()

    return value.casefold()


def infer_filename_from_url(url: str | None) -> str | None:
    if not url:
        return None
    try:
        parsed = urlparse(url)
        name = Path(unquote(parsed.path)).name
        if name.lower().endswith(".pdf"):
            return name
    except Exception:
        return None
    return None


def infer_filename_from_text(*texts: str) -> str | None:
    for text in texts:
        clean = strip_ansi(text or "")
        for match in PDF_NAME_RE.finditer(clean):
            candidate = match.group(1).rstrip(".,);]")
            name = Path(unquote(candidate)).name
            if name.lower().endswith(".pdf"):
                return name
    return None


def finalize_source_entry(index: int, lines: list[str]) -> dict[str, Any]:
    raw = normalize_spaces("\n".join(lines))

    explicit_filename = _extract_structured_field(raw, "filename", "title")
    explicit_title = _extract_structured_field(raw, "title", "pages")
    explicit_pages = _extract_structured_field(raw, "pages", "url")
    explicit_url = _extract_structured_field(raw, "url", None)

    page_numbers: list[int] = []
    if explicit_pages:
        for part in [p.strip() for p in explicit_pages.split(",") if p.strip()]:
            page_numbers.extend(parse_page_token(part))

    if not page_numbers:
        page_numbers = extract_page_numbers(raw)

    title = explicit_title or (lines[0].strip() if lines else "")
    title = re.sub(r"\s+", " ", title).strip()

    filename = (
        explicit_filename
        or infer_filename_from_text(raw, title)
        or infer_filename_from_url(explicit_url)
    )

    return {
        "index": index,
        "title": title,
        "raw": raw,
        "url": explicit_url,
        "page_numbers": sorted(set(page_numbers)),
        "filename": filename,
        "filename_normalized": normalize_filename(filename),
    }


def parse_sources(stdout_text: str) -> list[dict[str, Any]]:
    text = normalize_spaces(stdout_text)
    if not text:
        return []

    lines = text.split("\n")
    in_sources = False
    entries: list[dict[str, Any]] = []
    current_index: int | None = None
    current_lines: list[str] = []

    for line in lines:
        if SOURCES_MARKER_RE.match(line):
            in_sources = True
            continue

        if not in_sources:
            continue

        match = SOURCE_START_RE.match(line)
        if match:
            if current_index is not None:
                entries.append(finalize_source_entry(current_index, current_lines))
            current_index = int(match.group(1) or match.group(2))
            rest = (match.group(3) or "").strip()
            current_lines = [rest] if rest else []
            continue

        if current_index is not None:
            current_lines.append(line)

    if current_index is not None:
        entries.append(finalize_source_entry(current_index, current_lines))

    return entries


def parse_answer_citations(answer: str) -> list[int]:
    return sorted({int(x) for x in CITATION_RE.findall(answer or "")})


def run_ask_command(
    repo_root: Path,
    ask_module: str,
    cli_command: str,
    question: str,
    k: int,
    timeout_sec: int,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        ask_module,
        cli_command,
        question,
        "--k",
        str(k),
    ]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    started_at = now_utc_iso()
    t0 = time.perf_counter()

    try:
        completed = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            env=env,
            check=False,
        )
        duration_sec = round(time.perf_counter() - t0, 4)

        return {
            "status": "ok" if completed.returncode == 0 else "cli_error",
            "started_at_utc": started_at,
            "finished_at_utc": now_utc_iso(),
            "duration_sec": duration_sec,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "command": cmd,
            "command_str": subprocess.list2cmdline(cmd),
        }

    except subprocess.TimeoutExpired as e:
        duration_sec = round(time.perf_counter() - t0, 4)
        return {
            "status": "timeout",
            "started_at_utc": started_at,
            "finished_at_utc": now_utc_iso(),
            "duration_sec": duration_sec,
            "returncode": None,
            "stdout": e.stdout or "",
            "stderr": e.stderr or f"Timeout after {timeout_sec} sec",
            "command": cmd,
            "command_str": subprocess.list2cmdline(cmd),
        }


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

def extract_gold_references(item: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []

    primary_reference = item.get("primary_reference") or {}
    if not isinstance(primary_reference, dict):
        primary_reference = {}

    primary_filename = primary_reference.get("source_filename", item.get("source_filename"))
    primary_pages = safe_int_list(
        primary_reference.get("source_pages", item.get("source_pages"))
    )
    primary_gold_answer = str(
        primary_reference.get("gold_answer", item.get("gold_answer", ""))
    ).strip()

    refs.append(
        {
            "ref_kind": "primary",
            "source_filename": str(primary_filename or "").strip(),
            "source_pages": primary_pages,
            "gold_answer": primary_gold_answer,
        }
    )

    alternative_references = item.get("alternative_references") or []
    if isinstance(alternative_references, list):
        for alt in alternative_references:
            if not isinstance(alt, dict):
                continue

            alt_filename = str(alt.get("source_filename") or "").strip()
            alt_pages = safe_int_list(alt.get("source_pages"))
            alt_gold_answer = str(alt.get("gold_answer", "")).strip()

            refs.append(
                {
                    "ref_kind": "alternative",
                    "source_filename": alt_filename,
                    "source_pages": alt_pages,
                    "gold_answer": alt_gold_answer,
                }
            )

    cleaned: list[dict[str, Any]] = []
    for ref in refs:
        filename = str(ref.get("source_filename") or "").strip()
        pages = safe_int_list(ref.get("source_pages"))
        cleaned.append(
            {
                "ref_kind": str(ref.get("ref_kind") or "").strip(),
                "source_filename": filename,
                "source_pages": pages,
                "source_filename_normalized": normalize_filename(filename),
                "gold_answer": str(ref.get("gold_answer") or "").strip(),
            }
        )

    return cleaned


def evaluate_one_question(
    item: dict[str, Any],
    repo_root: Path,
    ask_module: str,
    cli_command: str,
    k: int,
    timeout_sec: int,
    run_id: str,
) -> dict[str, Any]:
    question_id = str(item.get("question_id", "")).strip() or "unknown_question_id"
    question = str(item.get("question", "")).strip()
    question_type = str(item.get("question_type", "")).strip()
    expected_answerable = bool(item.get("answerable", False))
    gold_answer = str(item.get("gold_answer", "")).strip()

    primary_reference = item.get("primary_reference") or {}
    if not isinstance(primary_reference, dict):
        primary_reference = {}

    gold_source_pages = safe_int_list(
        primary_reference.get("source_pages", item.get("source_pages"))
    )
    source_filename = primary_reference.get(
        "source_filename",
        item.get("source_filename"),
    )

    notes = item.get("notes") or item.get("comment")
    gold_references = extract_gold_references(item)

    if not question:
        return {
            "run_id": run_id,
            "question_id": question_id,
            "question": question,
            "question_type": question_type,
            "expected_answerable": expected_answerable,
            "gold_answer": gold_answer,
            "gold_source_pages": gold_source_pages,
            "source_filename": source_filename,
            "notes": notes,
            "status": "invalid_input",
            "error": "Пустой question",
        }

    exec_result = run_ask_command(
        repo_root=repo_root,
        ask_module=ask_module,
        cli_command=cli_command,
        question=question,
        k=k,
        timeout_sec=timeout_sec,
    )

    stdout = exec_result.get("stdout", "") or ""
    stderr = exec_result.get("stderr", "") or ""

    model_answer = parse_answer(stdout)
    parsed_sources = parse_sources(stdout)
    answer_citations = parse_answer_citations(model_answer)

    if answer_citations:
        active_sources = [src for src in parsed_sources if src.get("index") in answer_citations]
    else:
        active_sources = list(parsed_sources)

    all_cited_pages_set: set[int] = set()
    for src in active_sources:
        all_cited_pages_set.update(src.get("page_numbers", []))
    cited_source_pages_all = sorted(all_cited_pages_set)

    gold_source_filename_norm = normalize_filename(source_filename)

    matching_gold_doc_sources = [
        src
        for src in active_sources
        if src.get("filename_normalized") == gold_source_filename_norm
    ]

    cited_gold_doc_pages_set: set[int] = set()
    for src in matching_gold_doc_sources:
        cited_gold_doc_pages_set.update(src.get("page_numbers", []))
    cited_source_pages = sorted(cited_gold_doc_pages_set)

    cited_source_indexes_all = sorted(
        {int(src["index"]) for src in active_sources if "index" in src}
    )
    cited_source_indexes_gold_doc = sorted(
        {int(src["index"]) for src in matching_gold_doc_sources if "index" in src}
    )

    gold_pages_set = set(gold_source_pages)
    cited_pages_set_source_aware = set(cited_source_pages)
    cited_pages_set_legacy = set(cited_source_pages_all)

    page_hit_legacy = bool(gold_pages_set & cited_pages_set_legacy) if gold_pages_set else False
    all_gold_pages_covered_legacy = (
        bool(gold_pages_set) and gold_pages_set.issubset(cited_pages_set_legacy)
    )
    exact_cited_pages_match_legacy = (
        bool(gold_pages_set) and gold_pages_set == cited_pages_set_legacy
    )

    page_hit = bool(gold_pages_set & cited_pages_set_source_aware) if gold_pages_set else False

    all_gold_pages_covered = (
        bool(gold_pages_set) and gold_pages_set.issubset(cited_pages_set_source_aware)
    )
    exact_cited_pages_match = (
        bool(gold_pages_set) and gold_pages_set == cited_pages_set_source_aware
    )

    gold_source_match_found_any = False
    page_hit_any = False
    all_gold_pages_covered_any = False
    exact_cited_pages_match_any = False

    cited_source_indexes_any_gold_set: set[int] = set()
    cited_source_pages_any_gold_set: set[int] = set()
    matched_gold_references: list[dict[str, Any]] = []

    for ref in gold_references:
        ref_filename_norm = ref.get("source_filename_normalized") or ""
        ref_pages_set = set(safe_int_list(ref.get("source_pages")))

        if not ref_filename_norm:
            continue

        matching_sources_for_ref = [
            src
            for src in active_sources
            if src.get("filename_normalized") == ref_filename_norm
        ]

        if matching_sources_for_ref:
            gold_source_match_found_any = True

        cited_pages_for_ref: set[int] = set()
        for src in matching_sources_for_ref:
            if "index" in src:
                try:
                    cited_source_indexes_any_gold_set.add(int(src["index"]))
                except (TypeError, ValueError):
                    pass
            cited_pages_for_ref.update(src.get("page_numbers", []))

        if cited_pages_for_ref:
            cited_source_pages_any_gold_set.update(cited_pages_for_ref)

        if ref_pages_set and (ref_pages_set & cited_pages_for_ref):
            page_hit_any = True
            matched_gold_references.append(ref)

        if ref_pages_set and ref_pages_set.issubset(cited_pages_for_ref):
            all_gold_pages_covered_any = True

        if ref_pages_set and ref_pages_set == cited_pages_for_ref:
            exact_cited_pages_match_any = True

    cited_source_indexes_any_gold = sorted(cited_source_indexes_any_gold_set)
    cited_source_pages_any_gold = sorted(cited_source_pages_any_gold_set)

    predicted_has_answer = not is_refusal_answer(model_answer) if model_answer else False
    answerability_match = predicted_has_answer == expected_answerable

    record = {
        "run_id": run_id,
        "question_id": question_id,
        "question": question,
        "question_type": question_type,
        "expected_answerable": expected_answerable,
        "gold_answer": gold_answer,
        "gold_source_pages": gold_source_pages,
        "source_filename": source_filename,
        "gold_source_filename_normalized": gold_source_filename_norm,
        "notes": notes,
        "status": exec_result["status"],
        "started_at_utc": exec_result["started_at_utc"],
        "finished_at_utc": exec_result["finished_at_utc"],
        "duration_sec": exec_result["duration_sec"],
        "returncode": exec_result["returncode"],
        "command": exec_result["command"],
        "command_str": exec_result["command_str"],
        "model_answer": model_answer,
        "predicted_has_answer": predicted_has_answer,
        "refusal_match": is_refusal_answer(model_answer),
        "answerability_match": answerability_match,
        "answer_citations": answer_citations,
        "parsed_sources": parsed_sources,
        "cited_source_indexes_all": cited_source_indexes_all,
        "cited_source_indexes_gold_doc": cited_source_indexes_gold_doc,
        "cited_source_filenames_all": [
            src.get("filename") or "" for src in active_sources
        ],
        "debug_all_filenames": [
            src.get("filename_normalized") for src in parsed_sources
        ],
        "cited_source_filenames_gold_doc": [
            src.get("filename") or "" for src in matching_gold_doc_sources
        ],
        "gold_source_match_found": bool(matching_gold_doc_sources),
        "cited_source_pages_all": cited_source_pages_all,
        "cited_source_pages": cited_source_pages,
        "page_hit": page_hit,
        "all_gold_pages_covered": all_gold_pages_covered,
        "exact_cited_pages_match": exact_cited_pages_match,
        "page_hit_legacy": page_hit_legacy,
        "all_gold_pages_covered_legacy": all_gold_pages_covered_legacy,
        "exact_cited_pages_match_legacy": exact_cited_pages_match_legacy,
        "stdout_raw": stdout,
        "stderr_raw": stderr,
                "gold_references": gold_references,
        "gold_source_match_found_any": gold_source_match_found_any,
        "cited_source_indexes_any_gold": cited_source_indexes_any_gold,
        "cited_source_pages_any_gold": cited_source_pages_any_gold,
        "page_hit_any": page_hit_any,
        "all_gold_pages_covered_any": all_gold_pages_covered_any,
        "exact_cited_pages_match_any": exact_cited_pages_match_any,
        "matched_gold_references": matched_gold_references,
    }
    return record


def build_summary(
    records: list[dict[str, Any]],
    run_id: str,
    questions_path: Path,
    output_jsonl: Path,
    repo_root: Path,
    k: int,
    ask_module: str,
) -> dict[str, Any]:
    total = len(records)
    ok_like = [r for r in records if r.get("status") in {"ok", "cli_error"}]
    pure_ok = [r for r in records if r.get("status") == "ok"]
    errors = [r for r in records if r.get("status") not in {"ok", "cli_error"}]

    answerable = [r for r in ok_like if r.get("expected_answerable") is True]
    no_answer = [r for r in ok_like if r.get("expected_answerable") is False]

    answerability_match_count = sum(bool(r.get("answerability_match")) for r in ok_like)
    no_answer_correct = sum(not bool(r.get("predicted_has_answer")) for r in no_answer)
    citation_presence_count = sum(bool(r.get("answer_citations")) for r in ok_like)

    page_hit_count = sum(
        bool(r.get("page_hit_any", r.get("page_hit"))) for r in answerable
    )
    all_gold_pages_covered_count = sum(
        bool(r.get("all_gold_pages_covered_any", r.get("all_gold_pages_covered")))
        for r in answerable
    )
    exact_page_match_count = sum(
        bool(r.get("exact_cited_pages_match_any", r.get("exact_cited_pages_match")))
        for r in answerable
    )

    gold_source_match_found_count = sum(
        bool(r.get("gold_source_match_found_any", r.get("gold_source_match_found")))
        for r in answerable
    )

    page_hit_legacy_count = sum(bool(r.get("page_hit_legacy")) for r in answerable)
    all_gold_pages_covered_legacy_count = sum(
        bool(r.get("all_gold_pages_covered_legacy")) for r in answerable
    )
    exact_page_match_legacy_count = sum(
        bool(r.get("exact_cited_pages_match_legacy")) for r in answerable
    )

    def rate(num: int, den: int) -> float | None:
        if den == 0:
            return None
        return round(num / den, 4)

    summary = {
        "run_id": run_id,
        "created_at_utc": now_utc_iso(),
        "repo_root": str(repo_root),
        "questions_path": str(questions_path),
        "output_jsonl": str(output_jsonl),
        "ask_module": ask_module,
        "k": k,
        "total_questions": total,
        "status_counts": {
            "ok": len(pure_ok),
            "cli_error": len([r for r in records if r.get("status") == "cli_error"]),
            "timeout": len([r for r in records if r.get("status") == "timeout"]),
            "invalid_input": len([r for r in records if r.get("status") == "invalid_input"]),
            "other_errors": len(errors),
        },
        "metrics": {
            "answerability_accuracy_proxy": rate(answerability_match_count, len(ok_like)),
            "no_answer_accuracy": rate(no_answer_correct, len(no_answer)),
            "citation_presence_rate": rate(citation_presence_count, len(ok_like)),
            "gold_source_match_found_rate_on_answerable": rate(
                gold_source_match_found_count,
                len(answerable),
            ),
            "page_hit_rate_on_answerable": rate(page_hit_count, len(answerable)),
            "all_gold_pages_covered_rate_on_answerable": rate(
                all_gold_pages_covered_count,
                len(answerable),
            ),
            "exact_cited_pages_match_rate_on_answerable": rate(
                exact_page_match_count,
                len(answerable),
            ),
            "page_hit_rate_on_answerable_legacy": rate(page_hit_legacy_count, len(answerable)),
            "all_gold_pages_covered_rate_on_answerable_legacy": rate(
                all_gold_pages_covered_legacy_count,
                len(answerable),
            ),
            "exact_cited_pages_match_rate_on_answerable_legacy": rate(
                exact_page_match_legacy_count,
                len(answerable),
            ),
        },
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Прогоняет baseline ask по gold JSONL и сохраняет run в JSONL + summary JSON."
    )
    parser.add_argument(
        "--questions",
        default="data/evaluation/questions_gold/gold_questions_v3_pilot27_strict.jsonl",
        help="Путь до gold JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/evaluation/runs",
        help="Папка для сохранения результатов baseline-run.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Корень репозитория. Если не указан, скрипт попробует найти его автоматически.",
    )
    parser.add_argument(
        "--ask-module",
        default="src.apps.cli.main",
        help="Python module для CLI.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="top-k для ask.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=180,
        help="Таймаут одного вопроса в секундах.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Базовое имя файла run без расширения. По умолчанию генерируется автоматически.",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=0.0,
        help="Пауза между вопросами.",
    )
    parser.add_argument(
        "--cli-command",
        default="ask_hybrid",
        help="Команда CLI для запуска RAG.",
    )

    return parser.parse_args()

def make_default_run_base_name(questions_path: Path) -> str:
    stem = questions_path.stem
    stem = stem.removesuffix("_strict")
    return f"hybrid_rag_{stem}"

def main() -> None:
    args = parse_args()

    script_path = Path(__file__).resolve()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root(script_path)

    questions_path = resolve_path(args.questions, repo_root)
    output_dir = resolve_path(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not questions_path.exists():
        raise FileNotFoundError(f"Не найден файл вопросов: {questions_path}")

    questions = load_jsonl(questions_path)
    if not questions:
        raise ValueError(f"Файл вопросов пустой: {questions_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_run_name = args.run_name or f"{make_default_run_base_name(questions_path)}_bm25"
    run_id = f"{base_run_name}_{timestamp}"

    output_jsonl = output_dir / f"{run_id}.jsonl"
    output_summary = output_dir / f"{run_id}.summary.json"

    print(f"[eval] repo_root      = {repo_root}")
    print(f"[eval] questions      = {questions_path}")
    print(f"[eval] output_jsonl   = {output_jsonl}")
    print(f"[eval] output_summary = {output_summary}")
    print(f"[eval] total          = {len(questions)}")
    print()

    records: list[dict[str, Any]] = []

    with output_jsonl.open("w", encoding="utf-8") as out_f:
        for idx, item in enumerate(questions, start=1):
            question_id = str(item.get("question_id", f"q_{idx:03d}"))
            question = str(item.get("question", "")).strip()

            print(f"[{idx}/{len(questions)}] {question_id} :: {question}")

            record = evaluate_one_question(
                item=item,
                repo_root=repo_root,
                ask_module=args.ask_module,
                cli_command=args.cli_command,
                k=args.k,
                timeout_sec=args.timeout_sec,
                run_id=run_id,
            )

            records.append(record)
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            print(
                f"    status={record.get('status')} "
                f"duration={record.get('duration_sec')}s "
                f"predicted_has_answer={record.get('predicted_has_answer')} "
                f"gold_source_match_found={record.get('gold_source_match_found')} "
                f"page_hit={record.get('page_hit')} "
                f"page_hit_legacy={record.get('page_hit_legacy')}"
            )

            print(f"    answer={record.get('model_answer', '')}")

            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)

    summary = build_summary(
        records=records,
        run_id=run_id,
        questions_path=questions_path,
        output_jsonl=output_jsonl,
        repo_root=repo_root,
        k=args.k,
        ask_module=args.ask_module,
    )

    with output_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print("[eval] Готово.")
    print(f"[eval] JSONL   -> {output_jsonl}")
    print(f"[eval] SUMMARY -> {output_summary}")


if __name__ == "__main__":
    main()