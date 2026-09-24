# Integrated RAG System and Ablation

The default `POST /api/rag/query` pipeline keeps baseline vector retrieval and
uses reranking only when explicitly requested:

```text
Conversation memory (rewrite-only, when needed)
→ Query rewriting
→ ChromaDB retrieval
→ Native relevance distance
→ Evidence sufficiency check
→ Top-N evidence
→ Citation context
→ Grounded generation
```

The default request setting is `use_reranker=false`. Evidence is considered
usable when a non-empty text result has Chroma cosine distance at or below
`RAG_EVIDENCE_MAX_DISTANCE` (default `0.22`). The value is configurable by
environment variable and is reported in each RAG response. Results outside the
cutoff are not placed in the factual answer context; the API returns a Thai
insufficient-evidence instruction instead. The initial `0.22` value was chosen
from a smoke check separating one known in-dataset question (distance `0.134`)
from one unrelated question (distance `0.255`), not tuned on the final
evaluation output.

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
