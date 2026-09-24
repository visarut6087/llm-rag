Configure the existing RAG system to use this TXT file as the research source dataset:

/Users/foxixe/Downloads/Test llm /ragtest colab/gemini-ollama-gui/backend/dataset/sarabun 26.txt

Requirements:
- Treat this file as the current Thai article text dataset for the RAG research.
- Inspect the existing ingestion and dataset configuration before changing anything.
- Preserve the existing Vite + FastAPI + ChromaDB + embedding + Ollama architecture.
- Do not create, replace, or fabricate dataset content.
- Do not modify unrelated RAG features.
- Keep the original text unchanged.
- Ensure the path with spaces is handled correctly.
- Use the existing text chunking/ingestion mechanism unless there is a concrete bug preventing this file from being loaded.
- Preserve source metadata so retrieved chunks can later be traced back to:
  sarabun 26.txt
- Verify ingestion and retrieval using the actual file.
- Report:
  1. dataset path used
  2. number of text chunks ingested
  3. metadata/source name stored
  4. files changed
  5. verification result

Do not implement Relevance Score, Citation, Reranking, Query Rewriting, or other research phases yet.

Stop after dataset configuration and verification.