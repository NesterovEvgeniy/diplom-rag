"""загружает и обслуживает граф чанков для graph rag:
читает chunk_graph.jsonl, кэширует его в памяти,
позволяет получать соседей конкретного чанка
и проверять наличие чанка в графе."""


from __future__ import annotations

import json

from functools import lru_cache
from pathlib import Path


GRAPH_PATH = Path("data/processed/graph/chunk_graph.jsonl")


@lru_cache(maxsize=1)
def load_chunk_graph(path: str = str(GRAPH_PATH)) -> dict[str, dict]:
    graph_path = Path(path)
    if not graph_path.exists():
        raise FileNotFoundError(f"chunk graph file not found: {graph_path}")

    graph: dict[str, dict] = {}

    with graph_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except Exception as e:
                raise ValueError(f"invalid graph jsonl at line {line_no}: {e}") from e

            chunk_id = str(row.get("chunk_id") or "").strip()
            if not chunk_id:
                continue

            neighbors = row.get("neighbors") or []
            if not isinstance(neighbors, list):
                neighbors = []

            graph[chunk_id] = {
                "chunk_id": chunk_id,
                "doc_id": str(row.get("doc_id") or ""),
                "title": str(row.get("title") or ""),
                "source_pdf": str(row.get("source_pdf") or ""),
                "page_start": row.get("page_start"),
                "page_end": row.get("page_end"),
                "neighbors": [str(x) for x in neighbors if str(x).strip()],
            }

    return graph


def get_neighbors(chunk_id: str, graph_path: str = str(GRAPH_PATH)) -> list[str]:
    graph = load_chunk_graph(graph_path)
    node = graph.get(chunk_id)
    if not node:
        return []

    return list(node.get("neighbors") or [])


def has_chunk(chunk_id: str, graph_path: str = str(GRAPH_PATH)) -> bool:
    graph = load_chunk_graph(graph_path)
    return chunk_id in graph