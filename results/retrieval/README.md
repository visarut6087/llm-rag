# Retrieval vs. Reranking Results

Phase 3 provides an API-level comparison:

- Retrieval-only baseline: send `{"prompt":"...", "n_results":3, "use_reranker":false}`.
- Retrieval plus reranking: send the same request with `"use_reranker":true` (the default).

The response records `initial_rank`, `initial_score`, `rerank_score`, and `final_rank` for each returned item. The API also reports `candidate_count`, `reranking_enabled`, and score semantics. Save representative JSON responses in this directory after running the system against a fixed query set; do not compare runs using different corpora or `n_results` values.

No quantitative measurements are included yet because the project does not currently contain a labeled retrieval evaluation set. Phase 6 will add the evaluation workflow.
