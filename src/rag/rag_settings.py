from dataclasses import dataclass

from src.settings import get_settings  # pyright: ignore[reportMissingImports]


@dataclass(frozen=True)
class RagSettings:
    mode: str
    strict_citations: bool
    hybrid_retrieval_mode: str
    hybrid_dense_k: int
    hybrid_sparse_k: int
    hybrid_rrf_k: int


def get_rag_settings() -> RagSettings:
    s = get_settings()
    return RagSettings(
        mode=s.RAG_MODE,
        strict_citations=s.STRICT_CITATIONS,
        hybrid_retrieval_mode=s.HYBRID_RETRIEVAL_MODE,
        hybrid_dense_k=s.HYBRID_DENSE_K,
        hybrid_sparse_k=s.HYBRID_SPARSE_K,
        hybrid_rrf_k=s.HYBRID_RRF_K,
    )