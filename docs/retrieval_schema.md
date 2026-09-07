# Retrieval Result Schema

Phase 1 exposes structured retrieval results from `POST /api/rag/query`.

Each text result contains:

| Field | Meaning |
|---|---|
| `document_id` | ChromaDB record ID |
| `source` | Ingestion source name |
| `source_id` | Stable hash-based identity for the source |
| `source_name` | Human-readable source name |
| `chunk_id` | Stable source-local chunk identifier; legacy records fall back to `document_id` |
| `retrieval_score` | The native ChromaDB distance returned for the query |
| `rank` | One-based result rank within the collection |
| `text` | Retrieved chunk text |

`content` and `distance` remain in the response for backward compatibility. `retrieval_score` is deliberately not renamed to similarity: both text and image collections use ChromaDB's configured cosine distance, where lower values indicate closer results. No distance-to-similarity conversion is applied.

Image results expose the same required fields, with the image filename in `text` and `name`, plus the existing base64 payload in `b64`.

For Phase 2, each query result also receives a query-local `citation_id` such as `S1`. The API returns a `sources` catalog containing only retrieved records. The RAG context labels evidence with these IDs, and the model is instructed to cite only those IDs. The frontend displays the retrieved source catalog and flags missing or unknown citations; it does not invent source records.

Phase 3 adds second-stage reranking. When `use_reranker` is true (the default), ChromaDB returns up to `n_results * 3` candidates, the reranker scores them, and the best `n_results` are sent to the LLM. Each result records `initial_rank`, `initial_score`, `rerank_score`, and `final_rank`. The default rerank score is a deterministic lexical-overlap score (token overlap weighted 0.6 plus character trigram overlap weighted 0.4), where higher is better. Set `use_reranker` to false to run the retrieval-only baseline for comparison.

Phase 4 exposes `original_query`, `rewritten_query`, `query_rewrite_latency_ms`, and `query_rewrite_method` in every RAG response. Retrieval embeddings and reranking use `rewritten_query`; the original question remains unchanged for answer generation and chat history. An optional `conversation_context` request field can be used for controlled rewrite experiments. It is used only to resolve follow-up references and is never included as factual evidence. If rewriting fails or no rewrite context is applicable, the original query is used unchanged.

Phase 5 connects the UI's previous chat turns to `conversation_context` (limited to the latest eight turns). The rewriter considers prior user turns for intent/coreference and ignores prior assistant messages. Conversation history is not added to the evidence block or source catalog; citations can only refer to retrieved records.

Newly ingested text chunks receive IDs such as `source#chunk-1`. Existing ChromaDB records created before Phase 1 may not have `chunk_id`; their result uses the ChromaDB record ID as the fallback.
