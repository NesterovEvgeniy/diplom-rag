"""содержит вспомогательные функции для работы с citations в ответах rag:
проверяет корректность ссылок на источники, сопоставляет предложения с чанками,
автоматически проставляет ссылки вида [n]
и перенумеровывает их по порядку первого появления."""


from __future__ import annotations

import re

_CIT_RE = re.compile(r"\[(\d+)\]")
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9-]+")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")


def _has_valid_citations(answer: str, n_sources: int) -> bool:
    for m in _CIT_RE.findall(answer or ""):
        try:
            i = int(m)
        except ValueError:
            continue
        if 1 <= i <= n_sources:
            return True
    return False


def _normalize_match_text(text: str) -> str:
    text = (text or "").lower()
    text = _CIT_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize_match_text(text: str) -> set[str]:
    return {
        token
        for token in _WORD_RE.findall(_normalize_match_text(text))
        if len(token) >= 3
    }


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.findall((text or "").strip()) if s.strip()]


def _best_chunk_idx_for_sentence(sentence: str, chunks: list[dict]) -> int | None:
    sent_clean = _normalize_match_text(sentence)
    sent_tokens = _tokenize_match_text(sent_clean)

    if not sent_tokens:
        return None

    best_idx: int | None = None
    best_score = 0.0

    for idx, ch in enumerate(chunks):
        chunk_text = _normalize_match_text(ch.get("text", ""))
        chunk_tokens = _tokenize_match_text(chunk_text)

        if not chunk_tokens:
            continue

        overlap = len(sent_tokens & chunk_tokens) / max(1, len(sent_tokens))

        substring_boost = 0.0
        if sent_clean and sent_clean in chunk_text:
            substring_boost = 0.25

        score = overlap + substring_boost

        if score > best_score:
            best_score = score
            best_idx = idx

    if best_score < 0.18:
        return None

    return best_idx


def _assign_single_chunk_citations(answer: str, chunks: list[dict]) -> tuple[str, list[int]]:
    raw_sentences = _split_sentences(answer)
    if not raw_sentences:
        return "", []

    groups: list[dict[str, object]] = []

    for sentence in raw_sentences:
        clean_sentence = _normalize_match_text(sentence)
        if not clean_sentence:
            continue

        chunk_idx = _best_chunk_idx_for_sentence(clean_sentence, chunks)

        if groups and groups[-1]["chunk_idx"] == chunk_idx:
            groups[-1]["sentences"].append(sentence.strip())
        else:
            groups.append(
                {
                    "chunk_idx": chunk_idx,
                    "sentences": [sentence.strip()],
                }
            )

    result_parts: list[str] = []
    used_chunk_idxs: list[int] = []

    for group in groups:
        chunk_idx = group["chunk_idx"]
        sentences = group["sentences"]
        block_text = " ".join(sentences).strip()
        block_text = _CIT_RE.sub("", block_text)
        block_text = re.sub(r"\s+([.,!?;:])", r"\1", block_text)
        block_text = re.sub(r"\s+", " ", block_text).strip()

        if chunk_idx is not None:
            result_parts.append(f"{block_text} [{chunk_idx + 1}]")
            if chunk_idx not in used_chunk_idxs:
                used_chunk_idxs.append(chunk_idx)
        else:
            result_parts.append(block_text)

    final_text = " ".join(result_parts).strip()
    return final_text, used_chunk_idxs


def remap_citations_by_first_appearance(answer: str, n_sources: int) -> tuple[str, list[int]]:
    text = answer or ""

    raw_nums = [
        int(x)
        for x in _CIT_RE.findall(text)
        if x.isdigit() and 1 <= int(x) <= n_sources
    ]

    ordered_old_nums: list[int] = []
    for num in raw_nums:
        if num not in ordered_old_nums:
            ordered_old_nums.append(num)

    if not ordered_old_nums:
        return text, []

    old_to_new = {old_num: new_num for new_num, old_num in enumerate(ordered_old_nums, start=1)}

    def _replace_single_citation(match: re.Match[str]) -> str:
        old_num = int(match.group(1))
        new_num = old_to_new.get(old_num)
        if new_num is None:
            return ""
        return f"[{new_num}]"

    remapped = _CIT_RE.sub(_replace_single_citation, text)

    def _normalize_cluster(match: re.Match[str]) -> str:
        nums = [int(x) for x in re.findall(r"\[(\d+)\]", match.group(0))]
        nums = sorted(set(nums))
        return "".join(f"[{n}]" for n in nums)

    remapped = re.sub(r"(?:\[\d+\]){2,}", _normalize_cluster, remapped)
    remapped = re.sub(r"\s+", " ", remapped).strip()

    used_chunk_idxs = [old_num - 1 for old_num in ordered_old_nums]
    return remapped, used_chunk_idxs