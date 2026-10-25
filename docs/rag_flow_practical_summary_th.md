# สรุป RAG Flow ที่ใช้งานจริง: จากข้อมูลเข้า ChromaDB ถึงการตอบด้วย Ollama LLM

เอกสารนี้สรุป flow ของโปรเจกต์ `gemini-ollama-gui` ตามโค้ดปัจจุบัน โดยครอบคลุมการนำเข้าข้อมูล, การสร้าง embedding, การค้นคืนจาก ChromaDB, การประกอบ RAG context และการส่ง context ให้ Ollama เพื่อสร้างคำตอบ

## 1. ภาพรวมสถาปัตยกรรม

```text
ข้อมูลต้นฉบับ
  ├─ ข้อความ / ไฟล์ .txt .md .json .csv
  └─ รูปภาพ / .png .jpg .jpeg .webp .bmp
          │
          ▼
FastAPI backend :8000
  ├─ แบ่งข้อความเป็น paragraph chunks
  ├─ multilingual-e5-large → text vector 768 มิติ
  └─ CLIP ViT-B/32 → image vector 512 มิติ
          │
          ▼
ChromaDB แบบ persistent: backend/chroma_db/
  ├─ collection: text_chunks
  └─ collection: images
          │
ผู้ใช้ถามผ่านหน้าเว็บ Vite :3000
          │
          ▼
query rewrite (ถ้าเป็นคำถามต่อเนื่อง)
  → embed query
  → ChromaDB top-k retrieval
  → optional lexical reranking
  → evidence cutoff + citation labels [S1], [S2], ...
  → context_str
  → Ollama /api/chat
  → คำตอบแบบ streaming กลับหน้าเว็บ
```

จุดสำคัญคือ ChromaDB ไม่ได้สร้างคำตอบเอง แต่ทำหน้าที่ค้นหา chunks ที่ใกล้กับคำถาม ส่วน Ollama เป็นผู้สร้างคำตอบโดยได้รับหลักฐานใน `context_str` เพิ่มเข้าไปในข้อความล่าสุด

## 2. การเตรียมและรันระบบ

ติดตั้ง dependency ฝั่ง Python และ JavaScript:

```bash
cd "/Users/foxixe/Downloads/Test llm /ragtest colab/gemini-ollama-gui"
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
npm install
```

เปิด Ollama และตรวจสอบว่ามีโมเดลที่ต้องการ:

```bash
ollama serve
ollama list
ollama run qwen2.5:7b
```

เปิด backend และ frontend คนละ terminal:

```bash
# Terminal 1
cd "/Users/foxixe/Downloads/Test llm /ragtest colab/gemini-ollama-gui"
source .venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Terminal 2
npm run dev
```

เข้าใช้งานที่ `http://localhost:3000` โดย Vite จะ proxy `/api/ingest/*` และ `/api/rag/*` ไปยัง FastAPI ส่วน `/api/chat` และ `/api/tags` ไปยัง Ollama

หมายเหตุ: ครั้งแรกที่ backend เริ่มทำงาน อาจต้องดาวน์โหลด embedding models `intfloat/multilingual-e5-large` และ `clip-ViT-B-32` จึงใช้เวลานานกว่าครั้งถัดไป

## 3. Flow การนำข้อมูลเข้า ChromaDB

### 3.1 นำเข้าข้อความจาก dataset folder

ไฟล์ใน `backend/dataset/` จะถูกสแกนโดย `process_dataset_folder()` ใน `backend/ingest_dataset.py` รองรับ `.txt`, `.md`, `.json`, `.csv` และรูปภาพตามนามสกุลที่กำหนด

```python
if ext in SUPPORTED_TEXT_EXTS:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().strip()

    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    for index, p in enumerate(paragraphs, start=1):
        vec = embedder.embed_text(p)
        db.add_text(
            text=p,
            embedding=vec,
            source_name=file,
            chunk_id=f"{file}#chunk-{index}",
        )
```

ลำดับการทำงานคืออ่านไฟล์ → แบ่งด้วยบรรทัดว่าง → สร้าง vector ต่อ paragraph → เก็บ vector, เนื้อหา และ metadata ลง collection `text_chunks`

เรียกผ่าน API ได้ดังนี้:

```bash
curl -X POST http://localhost:8000/api/ingest/dataset \
  -H 'Content-Type: application/json' \
  -d '{}'
```

หรือเรียก Python โดยตรง:

```bash
python backend/ingest_dataset.py
```

### 3.2 นำเข้าข้อความจาก UI/API

ข้อความจะถูกแบ่งด้วย `\n\n` เช่นเดียวกัน:

```bash
curl -X POST http://localhost:8000/api/ingest/text \
  -H 'Content-Type: application/json' \
  -d '{
    "source_name": "manual-test",
    "text": "ChromaDB เป็นฐานข้อมูลเวกเตอร์\n\nRAG ใช้ข้อมูลที่ค้นคืนมาเป็นหลักฐาน"
  }'
```

ผลลัพธ์จะระบุจำนวน chunks ที่เพิ่ม เช่น `{"status":"success","chunks_added":2,...}`

### 3.3 การสร้าง embedding

ไฟล์ `backend/embedder.py` ใช้โมเดลคนละตัวสำหรับข้อความและรูปภาพ:

```python
def embed_text(self, text: str) -> list[float]:
    embedding = self.text_model.encode(
        f"passage: {text}", normalize_embeddings=True
    )
    return embedding.tolist()

def embed_query(self, query: str) -> list[float]:
    embedding = self.text_model.encode(
        f"query: {query}", normalize_embeddings=True
    )
    return embedding.tolist()
```

`multilingual-e5-large` ใช้ prefix `passage:` ตอนเก็บเอกสาร และ `query:` ตอนค้นหา เพื่อให้ embedding อยู่ในรูปแบบที่โมเดลรองรับได้ดี โดยเฉพาะงานหลายภาษา/ภาษาไทย และ vector มี 768 มิติ

รูปภาพใช้ `clip-ViT-B-32` ขนาด 512 มิติ ส่วนคำถามจะถูก encode ใน CLIP space ด้วย `embed_image_query()` เพื่อค้นหารูปภาพที่เกี่ยวข้องกับข้อความ

### 3.4 การเก็บลง ChromaDB

`backend/db.py` สร้างฐานข้อมูลแบบ persistent ที่ `backend/chroma_db/` และมีสอง collection:

```python
self.client = chromadb.PersistentClient(path=DB_DIR)
self.text_collection = self.client.get_or_create_collection(
    name="text_chunks",
    metadata={"hnsw:space": "cosine"}
)
self.image_collection = self.client.get_or_create_collection(
    name="images",
    metadata={"hnsw:space": "cosine"}
)
```

ข้อมูลข้อความแต่ละรายการประกอบด้วย:

- `id`: UUID ของ record ใน ChromaDB
- `embedding`: vector ของ paragraph
- `document`: ข้อความจริงของ chunk
- `metadata.source_name`: ชื่อไฟล์หรือชื่อแหล่งข้อมูล
- `metadata.source_id`: hash ที่คงที่ของชื่อแหล่งข้อมูล
- `metadata.chunk_id`: เช่น `sarabun 26.txt#chunk-1`
- `metadata.type`: `text`

ตรวจสอบจำนวนข้อมูลได้ด้วย:

```bash
curl http://localhost:8000/api/rag/stats
```

## 4. Flow การถามคำถามด้วย RAG

### ขั้นที่ 1: Frontend ส่งคำถามไป RAG API

เมื่อเปิดปุ่ม RAG ในหน้าเว็บ `src/app.js` จะเรียก:

```javascript
const rRes = await fetch('/api/rag/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prompt: text,
    n_results: 3,
    conversation_context: chat.messages.slice(0, -1).slice(-8)
  })
});
```

`conversation_context` ใช้ช่วยทำความเข้าใจคำถามต่อเนื่องเท่านั้น ไม่ถือเป็นหลักฐานข้อเท็จจริงสำหรับคำตอบ

### ขั้นที่ 2: Rewrite query แบบปลอดภัย

ใน `backend/query_rewriter.py` ถ้าคำถามมีคำอ้างอิง เช่น “เรื่องนี้”, “ดังกล่าว”, “เมื่อไหร่” และมี user turn ก่อนหน้า ระบบจะรวมคำถามก่อนหน้ากับคำถามปัจจุบันเพื่อใช้ค้นหา:

```python
rewritten = (
    f"บริบทคำถามก่อนหน้า: {turns[-1]} "
    f"คำถามติดตาม: {query}"
)
```

การ rewrite นี้ไม่สร้างข้อเท็จจริงใหม่ และ assistant messages จะไม่ถูกใช้เป็นหลักฐาน

### ขั้นที่ 3: สร้าง query embedding และค้นหา top-k

ใน `backend/main.py` ระบบสร้าง embedding สำหรับข้อความและรูปภาพ แล้วเรียก `db.search_all()`:

```python
query_vec = embedder.embed_query(rewritten_query)
image_query_vec = embedder.embed_image_query(rewritten_query)

final_k = max(req.n_results or 3, 1)
candidate_k = final_k * 3 if req.use_reranker else final_k

results = db.search_all(
    query_embedding=query_vec,
    image_query_embedding=image_query_vec,
    n_results=candidate_k,
)
```

ChromaDB ใช้ cosine distance โดยค่าต่ำกว่าหมายถึงใกล้กว่า ระบบยังคงค่าเดิมไว้ใน `retrieval_score` ไม่ได้แปลงเป็น similarity

### ขั้นที่ 4: Optional reranking

ถ้า request ตั้ง `use_reranker=true` ระบบจะดึง candidate มากขึ้นก่อน แล้วจัดอันดับซ้ำด้วย lexical overlap ใน `backend/reranker.py`:

```python
return round((0.6 * token_score) + (0.4 * ngram_score), 6)
```

คะแนน rerank สูงกว่าหมายถึงตรงกับ query มากกว่า โดยใช้ token overlap 60% และ character trigram overlap 40% แล้วคืน top `n_results` สุดท้าย

### ขั้นที่ 5: ตรวจว่า evidence ใช้ได้หรือไม่

ระบบจะไม่นำทุกผลลัพธ์เข้า prompt แต่เลือกเฉพาะ text ที่มีข้อความและ cosine distance ไม่เกินค่า cutoff:

```python
def is_usable_text_evidence(result: dict, max_distance: float) -> bool:
    text = (result.get("text") or result.get("content") or "").strip()
    distance = result.get("retrieval_score")
    return (
        bool(text)
        and isinstance(distance, (int, float))
        and distance <= max_distance
    )
```

ค่าเริ่มต้นคือ `RAG_EVIDENCE_MAX_DISTANCE=0.22` และปรับได้ด้วย environment variable:

```bash
export RAG_EVIDENCE_MAX_DISTANCE=0.22
```

ถ้าไม่มี evidence ที่ผ่าน cutoff ระบบจะประกอบ context ให้โมเดลตอบว่า `ไม่พบข้อมูลที่เพียงพอจากแหล่งข้อมูล` แทนการเดา

### ขั้นที่ 6: สร้าง citation context

ผลลัพธ์แต่ละรายการจะได้รับ citation ID เฉพาะ query เช่น `[S1]`, `[S2]`:

```python
for result in result_list:
    citation_id = f"S{len(retrieved_sources) + 1}"
    result["citation_id"] = citation_id
```

จากนั้นระบบสร้าง `context_str` ลักษณะนี้:

```text
=== ข้อมูลจากคลังความรู้ (RAG Context) ===
[S1] source_id=source-... source_name=sarabun 26.txt
document_id=... chunk_id=sarabun 26.txt#chunk-3
<ข้อความจาก chunk>
========================================
ตอบโดยอิงจากหลักฐานด้านบนเท่านั้น หากไม่พอให้ตอบว่า
"ไม่พบข้อมูลที่เพียงพอจากแหล่งข้อมูล"
```

กติกานี้ทำให้ citation เป็น query-local label และโมเดลอ้างได้เฉพาะแหล่งที่ backend ส่งมาใน `sources`

## 5. จาก RAG context ไปยัง Ollama LLM

หลังจากได้ `context_str` แล้ว `src/app.js` จะต่อ context เข้าไปท้ายข้อความ user ล่าสุด:

```javascript
const messagesForApi = chat.messages.map((m, i) => {
  if (i === chat.messages.length - 1) {
    let appended = m.content;
    if (ragContext) appended += ragContext;
    return { ...m, content: appended };
  }
  return m;
});
```

จากนั้น `src/ollama.js` ส่งข้อความทั้งหมดไป `POST /api/chat`:

```javascript
body: JSON.stringify({
  model,
  messages: chatPayload,
  stream: true,
  options: { temperature, num_ctx: numCtx }
})
```

Ollama ส่งผลลัพธ์แบบ NDJSON streaming หน้าเว็บจึงทยอยแสดงคำตอบทีละ token/chunk และเมื่อจบจะเก็บข้อความพร้อม `sources` ลง `localStorage`

## 6. ทดลองใช้งานแบบ end-to-end

1. วางไฟล์ความรู้ไว้ใน `backend/dataset/` หรือใช้ API `/api/ingest/text`
2. เรียก ingest และตรวจจำนวนข้อมูล:

```bash
curl -X POST http://localhost:8000/api/ingest/dataset -H 'Content-Type: application/json' -d '{}'
curl http://localhost:8000/api/rag/stats
```

3. ทดลอง retrieval โดยตรง:

```bash
curl -s -X POST http://localhost:8000/api/rag/query \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "ระบบนี้ใช้ embedding model อะไร",
    "n_results": 3,
    "use_reranker": false,
    "use_query_rewriting": true,
    "use_citations": true
  }'
```

ให้ตรวจ field สำคัญเหล่านี้:

- `evidence_sufficient`: มีหลักฐานที่ผ่าน cutoff หรือไม่
- `evidence_count`: จำนวน chunks ที่นำไปใช้เป็นหลักฐาน
- `texts`: ผลการค้นคืนพร้อม `retrieval_score`, `source_name`, `chunk_id`
- `context_str`: prompt context ที่จะส่งให้ LLM
- `sources`: catalog ของ citation ที่ใช้ได้จริง
- `rewritten_query`: query ที่ใช้ค้นหา
- `pipeline`: ลำดับขั้นตอนที่ทำงานจริง

4. ทดลองผ่าน UI เปิดปุ่ม RAG แล้วถามคำถามเดิม ระบบจะค้นหา ChromaDB ก่อน จากนั้นนำ context ไปให้ Ollama สร้างคำตอบ
5. เปรียบเทียบ retrieval-only กับ reranking:

```bash
curl -s -X POST http://localhost:8000/api/rag/query \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"ระบบนี้ใช้ embedding model อะไร","n_results":3,"use_reranker":false}'

curl -s -X POST http://localhost:8000/api/rag/query \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"ระบบนี้ใช้ embedding model อะไร","n_results":3,"use_reranker":true}'
```

เมื่อเปิด reranker ให้ดู `initial_rank`, `initial_score`, `rerank_score` และ `final_rank`

## 7. สรุปโค้ดสำคัญแบบสั้น

| ไฟล์ | หน้าที่ |
|---|---|
| `backend/ingest_dataset.py` | scan dataset, แบ่ง chunks, เรียก embedding และบันทึก DB |
| `backend/embedder.py` | โหลด text/image embedding models และกำหนด prefix `passage:`/`query:` |
| `backend/db.py` | สร้าง ChromaDB collections, add records และ vector search |
| `backend/query_rewriter.py` | ขยายคำถามต่อเนื่องโดยไม่สร้างหลักฐานปลอม |
| `backend/reranker.py` | จัดอันดับ candidate ซ้ำด้วย lexical token/ngram overlap |
| `backend/main.py` | orchestration ของ query, cutoff, citation และ response schema |
| `src/app.js` | เปิด RAG, เรียก API, ต่อ `context_str` กับ user prompt และแสดง sources |
| `src/ollama.js` | ส่ง chat ไป Ollama และอ่าน streaming response |

## 8. ข้อควรระวังของ implementation ปัจจุบัน

- Chunking ยังเป็นการ split ด้วยบรรทัดว่าง ไม่มี token limit หรือ overlap
- การ ingest ซ้ำไฟล์เดิมจะเพิ่ม records ซ้ำ เพราะยังไม่มี deduplication/upsert
- `retrieval_score` คือ cosine distance: ค่าต่ำดีกว่า ไม่ใช่ similarity: ค่าสูงดีกว่า
- รูปภาพถูกค้นคืนแยก collection จากข้อความ ส่วน factual evidence cutoff ใช้กับ text results
- การตอบแบบ grounded เกิดจาก instruction ใน `context_str` และการคัด evidence ไม่ใช่การบังคับระดับ model
- ต้องรัน Ollama, FastAPI และ Vite ให้ครบจึงจะเห็น flow ตั้งแต่ ingest ถึงคำตอบใน UI

## 9. Prompt สำหรับให้ AI สร้าง flow นี้

ใช้ prompt ต่อไปนี้กับ AI coding assistant เพื่อสร้างระบบลักษณะเดียวกัน:

```text
สร้างระบบ Local Multimodal RAG สำหรับภาษาไทย โดยใช้ FastAPI เป็น backend,
ChromaDB เป็น persistent vector database, SentenceTransformers
`intfloat/multilingual-e5-large` สำหรับ embedding ข้อความ และ
`clip-ViT-B-32` สำหรับ image embedding/cross-modal image search
พร้อมเชื่อมต่อ Ollama สำหรับสร้างคำตอบแบบ streaming

ข้อกำหนดหลัก:
1. รองรับ ingest ข้อความจาก API และ scan โฟลเดอร์ dataset โดยรองรับ .txt, .md,
   .json, .csv; แบ่งข้อความเป็น chunks พร้อม source_name และ chunk_id
2. รองรับ ingest รูปภาพ .png, .jpg, .jpeg, .webp, .bmp แยก collection จากข้อความ
3. ใช้ `PersistentClient` ของ ChromaDB เก็บข้อมูลในโฟลเดอร์ local และตั้งค่า
   `hnsw:space=cosine`
4. ตอน ingest ข้อความให้ใช้ prefix `passage:` และตอน query ให้ใช้ prefix `query:`
   พร้อม normalize embeddings
5. ทำ endpoint `POST /api/rag/query` ที่มี prompt, n_results, use_reranker,
   use_query_rewriting, use_citations และ conversation_context
6. ใช้ conversation_context เฉพาะเพื่อแก้คำถามต่อเนื่อง ห้ามใช้ conversation history
   เป็น factual evidence
7. ทำ retrieval จาก ChromaDB แล้ว optional rerank candidate ด้วย lexical token overlap
   และ character trigram overlap พร้อมคืน initial_rank, initial_score,
   rerank_score, final_rank
8. กรอง evidence ด้วย cosine-distance cutoff ที่ตั้งผ่าน
   `RAG_EVIDENCE_MAX_DISTANCE` ค่าเริ่มต้น 0.22; ถ้าไม่มีหลักฐานให้ตอบว่า
   "ไม่พบข้อมูลที่เพียงพอจากแหล่งข้อมูล" และห้ามเดา
9. สร้าง query-local citation IDs เช่น [S1], [S2] และคืน sources catalog ที่ตรงกับ
   records ที่ค้นพบจริงเท่านั้น
10. คืน context_str ที่มีหลักฐาน, metadata, citation rule และ grounding rule
    เพื่อให้ frontend นำไปต่อท้าย user prompt ล่าสุดก่อนส่ง Ollama
11. สร้าง frontend ที่เปิด/ปิด RAG ได้, เรียก `/api/rag/query`, ต่อ context_str
    เข้ากับข้อความล่าสุด และแสดง sources/citations หลังคำตอบ
12. จัดทำ README หรือเอกสารภาษาไทยที่อธิบาย architecture, วิธีติดตั้ง, วิธีรัน,
    curl examples, flow ตั้งแต่ ingest ถึง answer และข้อจำกัดของระบบ

ส่งมอบเป็นโค้ดที่รันได้จริง แยกไฟล์ตามหน้าที่ พร้อมตรวจสอบ error กรณี prompt ว่าง,
collection ว่าง, embedding ล้มเหลว และ evidence ไม่เพียงพอ
```
