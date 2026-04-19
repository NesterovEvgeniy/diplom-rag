"""строит граф связей между чанками документов:
читает chunks.jsonl, связывает чанки внутри документа по соседству и страницам,
сохраняет граф соседей в jsonl
и формирует статистику по вершинам и рёбрам."""


from __future__ import annotations

import json

from collections import defaultdict
from pathlib import Path


CHUNKS_PATH = Path("data/processed/chunks/chunks.jsonl")
OUT_DIR = Path("data/processed/graph")
OUT_GRAPH_PATH = OUT_DIR / "chunk_graph.jsonl"
OUT_STATS_PATH = OUT_DIR / "graph_stats.json"


def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"chunks file not found: {path}")

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except Exception as e:
                raise ValueError(f"invalid json at line {line_no}: {e}") from e

            chunk_id = str(row.get("chunk_id") or "").strip()
            doc_id = str(row.get("doc_id") or "").strip()
            text = str(row.get("text") or "").strip()

            if not chunk_id or not doc_id or not text:
                continue

            page_start = row.get("page_start")
            page_end = row.get("page_end")
            page_num = row.get("page_num")

            if page_start is None:
                page_start = page_num
            if page_end is None:
                page_end = page_num

            row["page_start"] = int(page_start) if page_start is not None else None
            row["page_end"] = int(page_end) if page_end is not None else None
            rows.append(row)

    return rows


def add_edge(edge_map: dict[str, set[str]], src: str, dst: str) -> None:
    if not src or not dst or src == dst:
        return
    edge_map[src].add(dst)


def build_graph(chunks: list[dict]) -> tuple[list[dict], dict]:
    by_doc: dict[str, list[dict]] = defaultdict(list)
    by_doc_page: dict[tuple[str, int], list[dict]] = defaultdict(list)

    for row in chunks:
        doc_id = str(row["doc_id"])
        by_doc[doc_id].append(row)

        page_start = row.get("page_start")
        if isinstance(page_start, int):
            by_doc_page[(doc_id, page_start)].append(row)

    edge_map: dict[str, set[str]] = defaultdict(set)
    edge_type_counts = {
        "prev_next": 0,
        "same_page": 0,
        "adjacent_page": 0,
    }

    # 1. prev/next edges внутри документа
    for doc_id, doc_chunks in by_doc.items():
        doc_chunks_sorted = sorted(
            doc_chunks,
            key=lambda x: (
                int(x.get("page_start") or 10**9),
                str(x.get("chunk_id") or ""),
            ),
        )

        for i, row in enumerate(doc_chunks_sorted):
            cur_id = str(row["chunk_id"])

            if i > 0:
                prev_id = str(doc_chunks_sorted[i - 1]["chunk_id"])
                before = len(edge_map[cur_id])
                add_edge(edge_map, cur_id, prev_id)
                add_edge(edge_map, prev_id, cur_id)
                after = len(edge_map[cur_id])
                if after > before:
                    edge_type_counts["prev_next"] += 1

            if i + 1 < len(doc_chunks_sorted):
                next_id = str(doc_chunks_sorted[i + 1]["chunk_id"])
                before = len(edge_map[cur_id])
                add_edge(edge_map, cur_id, next_id)
                add_edge(edge_map, next_id, cur_id)
                after = len(edge_map[cur_id])
                if after > before:
                    edge_type_counts["prev_next"] += 1

    # 2. same_page edges
    for (doc_id, page), page_chunks in by_doc_page.items():
        ids = [str(x["chunk_id"]) for x in page_chunks]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                before_i = len(edge_map[ids[i]])
                before_j = len(edge_map[ids[j]])

                add_edge(edge_map, ids[i], ids[j])
                add_edge(edge_map, ids[j], ids[i])

                if len(edge_map[ids[i]]) > before_i or len(edge_map[ids[j]]) > before_j:
                    edge_type_counts["same_page"] += 1

    # 3. adjacent_page edges
    for row in chunks:
        doc_id = str(row["doc_id"])
        src_id = str(row["chunk_id"])
        page_start = row.get("page_start")

        if not isinstance(page_start, int):
            continue

        for near_page in (page_start - 1, page_start + 1):
            near_chunks = by_doc_page.get((doc_id, near_page), [])
            for near_row in near_chunks:
                dst_id = str(near_row["chunk_id"])
                before = len(edge_map[src_id])
                add_edge(edge_map, src_id, dst_id)
                if len(edge_map[src_id]) > before:
                    edge_type_counts["adjacent_page"] += 1

    graph_rows: list[dict] = []
    degrees: list[int] = []

    for row in chunks:
        chunk_id = str(row["chunk_id"])
        neighbors = sorted(edge_map.get(chunk_id, set()))
        degree = len(neighbors)
        degrees.append(degree)

        graph_rows.append(
            {
                "chunk_id": chunk_id,
                "doc_id": str(row["doc_id"]),
                "title": str(row.get("title") or ""),
                "source_pdf": str(row.get("source_pdf") or row.get("filename") or ""),
                "page_start": row.get("page_start"),
                "page_end": row.get("page_end"),
                "neighbors": neighbors,
            }
        )

    total_nodes = len(graph_rows)
    total_edges_directed = sum(len(r["neighbors"]) for r in graph_rows)
    avg_degree = (sum(degrees) / len(degrees)) if degrees else 0.0
    max_degree = max(degrees) if degrees else 0

    stats = {
        "total_nodes": total_nodes,
        "total_edges_directed": total_edges_directed,
        "avg_degree": round(avg_degree, 4),
        "max_degree": max_degree,
        "edge_type_counts": edge_type_counts,
    }

    return graph_rows, stats


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    chunks = load_chunks(CHUNKS_PATH)
    graph_rows, stats = build_graph(chunks)

    write_jsonl(OUT_GRAPH_PATH, graph_rows)
    write_json(OUT_STATS_PATH, stats)

    print(f"[graph] chunks_in      = {len(chunks)}")
    print(f"[graph] graph_out      = {OUT_GRAPH_PATH}")
    print(f"[graph] stats_out      = {OUT_STATS_PATH}")
    print(f"[graph] total_nodes    = {stats['total_nodes']}")
    print(f"[graph] total_edges    = {stats['total_edges_directed']}")
    print(f"[graph] avg_degree     = {stats['avg_degree']}")
    print(f"[graph] max_degree     = {stats['max_degree']}")


if __name__ == "__main__":
    main()