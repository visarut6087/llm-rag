# RAG Evaluation

Phase 6 evaluates retrieval independently from answer generation. The manually annotated dataset is:

`results/evaluation/rag_evaluation_dataset.json`

It is based on `backend/dataset/sarabun 26.txt`. Every item records `question`, `expected_answer`, `relevant_source`, `relevant_chunk`, and `answerable`; the unanswerable item explicitly uses null relevance labels rather than fabricated evidence.

Run the evaluator while the FastAPI backend is available:

```bash
python scripts/evaluate_rag.py --base-url http://localhost:8000
```

The evaluator preserves separate configurations for:

- `baseline_rag`
- `relevance_score`
- `reranking`
- `query_rewriting`
- `conversation_memory`

It computes Recall@K, Precision@K, Hit Rate@K, MRR, and nDCG@K for answerable items. It records retrieval, reranking, query-rewrite, and client total latency from each API response. Metrics for unanswerable items are null because there is no relevant source/chunk to retrieve.

Outputs:

- `results/evaluation/rag_evaluation.json`
- `results/evaluation/rag_evaluation.csv`

The current environment did not have the FastAPI, ChromaDB, or embedding packages installed when this phase was implemented, so the generated result files may have status `not_run`. No measurement values are fabricated; rerun the command after installing `backend/requirements.txt` and starting the backend.
