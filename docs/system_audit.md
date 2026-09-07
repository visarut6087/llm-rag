# System Audit

Audit phase: Phase 0  
Audit date: 2026-09-07  
Audited project: `ragtest colab/gemini-ollama-gui`

## Scope and assumptions

The workspace contains several archived/project-export directories. This audit targets `ragtest colab/gemini-ollama-gui` because it is the project containing the requested `src/`, `backend/`, Vite, FastAPI, Ollama, and ChromaDB components and its README describes the current multimodal RAG application.

This document records the implementation as inspected. No application behavior was changed during the audit.

## Architecture overview

```text
Browser (Vite frontend)
  ├─ Ollama chat/model APIs through Vite proxy or configured host
  ├─ FastAPI RAG/ingestion APIs through Vite proxy
  └─ DuckDuckGo Lite web-search middleware in Vite

FastAPI backend
  ├─ multilingual-e5-large text embeddings
  ├─ CLIP ViT-B/32 image and cross-modal embeddings
  └─ persistent ChromaDB collections: text_chunks, images
```

## Component classification

| Component | Status | Evidence / finding |
|---|---|---|
| Vite frontend | IMPLEMENTED | `src/app.js`, `src/ollama.js`, `src/style.css`, `index.html`, `vite.config.js` |
| FastAPI backend | IMPLEMENTED | `backend/main.py` exposes health, ingestion, stats, and RAG query endpoints |
| Ollama integration | IMPLEMENTED | `src/ollama.js` calls `/api/tags`, `/api/chat`, and `/api/pull`; responses stream as NDJSON |
| Text embeddings | IMPLEMENTED | `backend/embedder.py` loads `intfloat/multilingual-e5-large`; passage/query prefixes are used |
| Image embeddings | IMPLEMENTED | `backend/embedder.py` loads `clip-ViT-B-32` for images and text-to-image queries |
| ChromaDB storage | IMPLEMENTED | `backend/db.py` uses a persistent local client and two cosine collections |
| Text ingestion | IMPLEMENTED | Manual API input, dataset ingestion, paragraph splitting, embedding, and storage are present |
| Image ingestion | IMPLEMENTED | Upload and dataset paths embed images and store base64 data in metadata |
| Text chunking | PARTIAL | Splits only on blank lines; no token/character limits, overlap, stable chunk IDs, or document structure tracking |
| Text retrieval | IMPLEMENTED | ChromaDB top-k query returns documents, metadata, and distances internally |
| Image retrieval | IMPLEMENTED | Separate CLIP-space top-k query returns image name, base64 data, and distance |
| Retrieval score exposure | PARTIAL | Backend returns `distance` inside `texts`/`images`, but frontend context discards scores and no rank/document/chunk fields are exposed consistently |
| Retrieval citations/sources | MISSING | `source` metadata is stored for text but is not included in the assembled context or displayed in answers; no citation format exists |
| Reranking | MISSING | No reranker or second-stage relevance model was found |
| Query rewriting | MISSING | User prompt is sent directly to retrieval; no rewrite or multi-query step exists |
| Conversation memory to RAG | MISSING | Chat history is sent to Ollama, but retrieval uses only the current user text |
| Chat history | IMPLEMENTED | `localStorage` stores chats under `oc_chats`; messages are sent as Ollama history |
| Web search | IMPLEMENTED | Optional UI toggle calls `/api/search`; Vite middleware posts to DuckDuckGo Lite and returns up to three snippets |
| Existing metadata | PARTIAL | Text metadata has `source` and `type`; image metadata has `name`, base64 image, and `type`; no chunk index, document ID in result, timestamp, or provenance URL |
| RAG evaluation | MISSING | No labeled query set, retrieval metrics, test runner, or evaluation reports were found |
| LLM evaluation | MISSING | No answer-quality, faithfulness, citation, or hallucination evaluation capability was found |
| Ablation/final analysis | MISSING | No experiment configuration, run tracking, or ablation workflow was found |
| Authentication/authorization | MISSING | README notes there is no authentication or authorization layer |

## Current frontend flow

1. `src/app.js` loads settings and chat history from browser `localStorage`.
2. On boot it checks Ollama connectivity, fetches available models, restores the first chat, or creates a new chat.
3. A user message is appended to the active chat and rendered immediately.
4. If the RAG toggle is enabled, the frontend sends `{ prompt: text, n_results: 3 }` to `POST /api/rag/query`.
5. If web search is enabled, the frontend calls `/api/search?q=...`.
6. Retrieved RAG text and web-search snippets are appended to the latest user message as plain prompt text.
7. The resulting message history is sent to Ollama `/api/chat`; streamed assistant tokens update the UI and the completed answer is saved to `localStorage`.

RAG is opt-in from the UI and is not used for every chat by default.

## Current API flow

### RAG query

`POST /api/rag/query` performs the following:

1. Validate the prompt.
2. Embed the prompt with the multilingual text model using the `query:` prefix.
3. Embed the prompt in CLIP space for cross-modal image search.
4. Query the text and image ChromaDB collections independently.
5. Build `context_str` from text contents and image names.
6. Add an instruction requiring the model to answer only from the text context when text results exist.
7. Return the query, assembled context, text results, and image results.

The API currently returns text `content`, `metadata`, and `distance` internally through the result object, but the frontend uses only `context_str`; image results contain `name`, base64 data, and `distance`.

### Ingestion

- `POST /api/ingest/text`: splits input on double newlines, embeds each paragraph, and stores it in `text_chunks`.
- `POST /api/ingest/image`: embeds an uploaded image and stores it in `images`.
- `POST /api/ingest/dataset`: scans the configured dataset directory and processes supported text/image files.
- `GET /api/rag/stats`: returns text and image collection counts.

## Embedding and vector database details

- Text model: `intfloat/multilingual-e5-large`, documented as 768 dimensions.
- Text ingestion uses `passage: ...`; text queries use `query: ...`; embeddings are normalized.
- Image model: `clip-ViT-B-32`, documented as 512 dimensions.
- ChromaDB uses a persistent directory at `backend/chroma_db/`.
- Collections are `text_chunks` and `images`, both configured with cosine distance.
- Text IDs are generated UUIDs. Image IDs are also generated UUIDs.
- Text metadata: `source`, `type`.
- Image metadata: `name`, `b64`, `type`.

The implementation preserves ChromaDB's returned value as `distance`. The current code does not convert it to similarity and does not expose a documented score direction to the user.

## Chunking strategy

Text is split by `text.split("\\n\\n")`. Each non-empty paragraph becomes one vector entry. The same basic paragraph strategy is used by dataset ingestion. There is no explicit maximum chunk size, overlap, sentence segmentation, heading extraction, source document ID, or chunk index.

## Ollama request flow

`src/ollama.js` first tries Vite's relative proxy endpoints and falls back to the configured Ollama host. Chat requests include an optional system prompt, the stored chat message history, model name, temperature, and context length. The request uses `stream: true`; newline-delimited JSON response messages are parsed and forwarded to the UI incrementally.

## Chat history and memory

Chat records contain an ID, title, selected model, and message list. They are persisted in `localStorage` under `oc_chats`. This history is included in generation requests, but it is not used to construct the RAG query. Therefore conversational follow-up retrieval is not currently implemented.

## Web search

The frontend has an explicit web-search toggle. When enabled, Vite's `/api/search` middleware sends a POST request to DuckDuckGo Lite, parses result snippets, and returns at most three snippets. The snippets are appended to the latest user prompt without structured source URLs or citation metadata.

## Evaluation capability

No evaluation dataset, ground-truth answers, retrieval relevance labels, automated metrics, LLM-as-judge flow, experiment registry, or report-generation scripts were found in the audited project. The current system therefore cannot yet measure retrieval relevance, grounding, citation correctness, conversational retrieval, or hallucination reduction.

## Key Phase 1 readiness notes

The current retrieval layer already has the raw ChromaDB distance available, so a relevance-score phase can build on it. Before changing behavior, the following contracts should be defined:

- whether the exposed value is cosine distance or a derived similarity;
- stable fields for document ID, source, chunk ID, rank, text, and score;
- separate result schemas for text and image retrieval;
- whether scores are returned to the frontend and/or only used for evaluation;
- how empty collections and failed model/database requests are represented.

## Audit conclusion

The core local multimodal RAG application is IMPLEMENTED, including ingestion, embedding, vector search, optional RAG prompting, chat history, and optional web search. Research instrumentation is incomplete: provenance, standardized retrieval results, reranking, query rewriting, conversational retrieval, and all evaluation/ablation capabilities are absent or partial. Phase 0 is complete; application behavior was not modified.
