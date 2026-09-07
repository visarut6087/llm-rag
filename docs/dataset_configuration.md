# Dataset Configuration

## Active research dataset

The RAG dataset is the existing file:

`/Users/foxixe/Downloads/Test llm /ragtest colab/gemini-ollama-gui/backend/dataset/sarabun 26.txt`

The existing `backend/ingest_dataset.py` uses `backend/dataset/` as its default directory, scans `.txt` files, reads UTF-8 text, splits on blank lines, and stores each paragraph through the existing embedding and ChromaDB path. The filename `sarabun 26.txt` is passed as the source metadata, so the space in the filename is preserved without shell interpolation.

## Verification

- Original file content was not changed.
- SHA-256: `df6f210a47343aece7e668264f6955554f8b81e80d393efb374c503b3fe9228d`
- File size: 48,501 bytes.
- Existing chunking produces 44 non-empty chunks.
- The persisted ChromaDB contains 44 text embeddings in the `text_chunks` collection.
- Persisted metadata inspection found `source = "sarabun 26.txt"` on the text records.
- Existing records were metadata-migrated in place to add `source_id`, `source_name`, and `sarabun 26.txt#chunk-1` through `sarabun 26.txt#chunk-44`; the collection count remains 44 and no text/vector records were duplicated.

The backend dependencies were installed in the project-local `.venv`, and the FastAPI service was started successfully. `/` returned a healthy response, `/api/rag/stats` returned `text_count = 44` and `image_count = 0`, and a live RAG query returned source `sarabun 26.txt` with a valid `sarabun 26.txt#chunk-N` citation target. No duplicate re-ingestion was performed.

## Files changed for this request

- `docs/dataset_configuration.md` — records the existing configuration and verification.
- `backend/chroma_db/` — existing text-record metadata updated in place with stable source/chunk fields; no records were added or removed.

No application code, dataset content, or unrelated RAG feature was modified.
