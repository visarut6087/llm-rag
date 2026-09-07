# Final Research Analysis

## Scope

This report is the analysis template for the controlled A–G ablation in `results/ablation/ablation_results.json` and the retrieval/LLM evaluations in `results/evaluation/`.

## Required comparisons

| Configuration | Added component |
|---|---|
| A | Baseline RAG |
| B | Relevance-score instrumentation |
| C | Citations |
| D | Reranking |
| E | Query rewriting |
| F | Conversation memory for rewriting |
| G | Full RAG confirmation |

For each comparison, report retrieval quality, answer relevance, faithfulness, hallucination rate, citation accuracy, abstention correctness, and latency. Do not claim an improvement unless the corresponding paired measurements are present.

## Research questions

- RQ1: Does retrieval relevance improve evidence quality and reduce hallucination?
- RQ2: Does reranking improve final evidence ordering?
- RQ3: Does query rewriting improve Thai conversational retrieval?
- RQ4: Does conversation-aware rewriting help without turning conversation history into evidence?
- RQ5: Do citations improve factual traceability?
- RQ6: Which component contributes most to hallucination reduction?
- RQ7: What latency trade-offs accompany each component?

## Current status

The implementation and experiment runners are present. The local run remains `not_run` when FastAPI, ChromaDB, embedding models, or Ollama are unavailable. This report must preserve that negative result and must not substitute assumptions for measurements.
