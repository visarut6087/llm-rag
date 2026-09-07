"""Deterministic second-stage reranking for retrieved candidates.

This baseline reranker uses query/candidate lexical evidence rather than the
ChromaDB distance. It supports Thai text without requiring a word-segmentation
package by combining Unicode token overlap with character n-gram overlap.
"""

import re
from typing import Iterable


TOKEN_RE = re.compile(r"[\wก-๙]+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in TOKEN_RE.findall(text or "")}


def _ngrams(text: str, size: int = 3) -> set[str]:
    compact = re.sub(r"\s+", "", (text or "").casefold())
    if not compact:
        return set()
    if len(compact) <= size:
        return {compact}
    return {compact[index:index + size] for index in range(len(compact) - size + 1)}


def rerank_score(query: str, candidate: str) -> float:
    """Return a higher-is-better lexical relevance score in [0, 1]."""
    query_tokens = _tokens(query)
    candidate_tokens = _tokens(candidate)
    token_union = query_tokens | candidate_tokens
    token_score = len(query_tokens & candidate_tokens) / len(token_union) if token_union else 0.0

    query_ngrams = _ngrams(query)
    candidate_ngrams = _ngrams(candidate)
    ngram_union = query_ngrams | candidate_ngrams
    ngram_score = len(query_ngrams & candidate_ngrams) / len(ngram_union) if ngram_union else 0.0

    return round((0.6 * token_score) + (0.4 * ngram_score), 6)


def rerank_results(query: str, results: Iterable[dict], text_key: str = "text") -> list[dict]:
    """Add reranking metadata and return candidates in final rank order."""
    ranked = []
    for index, result in enumerate(results, start=1):
        item = dict(result)
        item["initial_rank"] = result.get("rank", index)
        item["initial_score"] = result.get("retrieval_score")
        item["rerank_score"] = rerank_score(query, result.get(text_key) or result.get("content", ""))
        ranked.append(item)

    ranked.sort(key=lambda item: (-item["rerank_score"], item["initial_rank"]))
    for final_rank, item in enumerate(ranked, start=1):
        item["final_rank"] = final_rank
        item["rank"] = final_rank
    return ranked
