# Integrated RAG System and Ablation

Phase 8 is enabled by the default `POST /api/rag/query` settings:

```text
Conversation memory (rewrite-only)
→ Query rewriting
→ ChromaDB retrieval
→ Native relevance distance
→ Reranking
→ Top-N evidence
→ Citation context
→ Grounded generation
```

The API returns the active `pipeline` and the explicit grounding rule:

> Use retrieved evidence as the factual basis; if evidence is insufficient, say so; do not invent unsupported facts.

Conversation turns are used only to understand the query. Retrieved records are the factual evidence and the only valid citation sources.

Phase 9 uses `scripts/run_ablation.py` to compare configurations A–G. The feature switches are request-level and preserve each baseline. Run:

```bash
python scripts/run_ablation.py --base-url http://localhost:8000
```

Outputs:

- `results/ablation/ablation_results.json`
- `results/ablation/ablation_results.csv`
- `docs/final_research_analysis.md`

Answer-quality fields are intentionally not inferred by the ablation runner; they must be populated from the Phase 7 LLM evaluation. Failed and unavailable runs remain in the output.
