# LLM Evaluation

Phase 7 evaluates generated answers separately from retrieval quality. The manually annotated dataset is:

`results/evaluation/llm_evaluation_dataset.json`

It includes answerable, unanswerable, ambiguous, multi-turn, and source-specific Thai questions. The unanswerable item explicitly requires abstention.

Run the evaluator with the RAG backend and Ollama available:

```bash
python scripts/evaluate_llm.py \
  --rag-url http://localhost:8000 \
  --ollama-url http://localhost:11434 \
  --model qwen2.5:7b
```

The script generates an answer from retrieved evidence, then asks the configured Ollama model to judge:

- faithfulness
- answer relevance
- context relevance
- citation correctness
- hallucination rate
- abstention correctness

These metrics are not substituted for retrieval metrics. Retrieval scores remain in `rag_evaluation.json`. The evaluator also records citation IDs found in the answer and whether the annotated source/chunk was cited.

Outputs:

- `results/evaluation/llm_evaluation.json`
- `results/evaluation/llm_evaluation.csv`

If either RAG or Ollama is unavailable, the output status is `not_run` or `partial` and no metric values are invented.
