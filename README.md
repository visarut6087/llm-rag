# Ollama Chat with Multimodal RAG

A local Thai-friendly chat interface for [Ollama](https://ollama.com/) with optional web search and multimodal retrieval-augmented generation (RAG).

The project contains:

- A Vite frontend with chat history, Markdown/code rendering, model selection, streaming responses, and settings.
- A FastAPI backend for text and image ingestion and ChromaDB vector search.
- Multilingual text embeddings using `intfloat/multilingual-e5-large`.
- Image and cross-modal text/image embeddings using `clip-ViT-B-32`.
- An optional bundled Qwen GGUF model and Ollama `Modelfile`.

## Requirements

- Node.js 18+ and npm
- Python 3.10+
- Ollama running locally at `http://localhost:11434`
- Enough disk space and RAM for the selected Ollama model and embedding models

The bundled `qwen2.5-3b-instruct.Q4_K_M.gguf` file is approximately 1.8 GB.

## Project layout

```text
.
├── src/                    # Vite frontend
├── index.html              # Application shell
├── vite.config.js          # Dev server and API proxies
├── backend/
│   ├── main.py             # FastAPI application
│   ├── embedder.py         # Text and image embedding models
│   ├── db.py               # ChromaDB storage and search
│   ├── ingest_dataset.py   # Dataset-folder ingestion
│   ├── dataset/            # Default text/image dataset location
│   └── requirements.txt
├── Modelfile
└── qwen2.5-3b-instruct.Q4_K_M.gguf
```

## Setup

### 1. Start Ollama

Install Ollama, start its server, and verify that it responds:

```bash
ollama list
```

To use the bundled GGUF model, run this from the project directory:

```bash
ollama create qwen2.5-3b-local -f Modelfile
ollama run qwen2.5-3b-local
```

Alternatively, pull any compatible model from the UI or with Ollama, for example:

```bash
ollama pull qwen2.5:7b
```

### 2. Install backend dependencies

```bash
cd backend
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Start the RAG API from the `backend` directory:

```bash
python main.py
```

The API listens on `http://localhost:8000`. The first startup downloads and loads the embedding models, so it may take a while.

### 3. Install and start the frontend

In a second terminal, from the project directory:

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

The Vite server proxies:

- Ollama requests to `http://localhost:11434`
- RAG and ingestion requests to `http://localhost:8000`
- Web-search requests through the local `/api/search` middleware

## Using the RAG knowledge base

The UI's **Knowledge Base (RAG)** button supports text and image ingestion. Enable the brain/RAG toggle before sending a chat message to retrieve relevant context.

To ingest the default dataset from the backend directory:

```bash
python ingest_dataset.py
```

Supported dataset files:

- Text: `.txt`, `.md`, `.json`, `.csv`
- Images: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`

Text is split on blank lines. ChromaDB data is stored in `backend/chroma_db/`.

## API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Health check |
| `GET` | `/api/rag/stats` | Return text/image counts |
| `POST` | `/api/ingest/text` | Add paragraph chunks to ChromaDB |
| `POST` | `/api/ingest/image` | Add an image embedding to ChromaDB |
| `POST` | `/api/ingest/dataset` | Scan and ingest a dataset folder |
| `POST` | `/api/rag/query` | Search text and image collections |

Example text ingestion:

```bash
curl -X POST http://localhost:8000/api/ingest/text \
  -H 'Content-Type: application/json' \
  -d '{"text":"นี่คือข้อมูลตัวอย่างสำหรับค้นคืน","source_name":"example"}'
```

## Frontend commands

```bash
npm run dev       # Start development server on port 3000
npm run build     # Create a production build in dist/
npm run preview   # Preview the production build
```

## Configuration

Use the settings panel in the UI to change:

- Ollama host URL
- System prompt
- Temperature
- Context length

Chat history and settings are stored in the browser's `localStorage`.

## Development notes

This is currently configured for local development. Before exposing it to a network or deploying it, review the permissive CORS policy in `backend/main.py`, the unrestricted dataset-folder input, the embedded base64 image storage, and the web-search proxy. The project has no authentication or authorization layer.

