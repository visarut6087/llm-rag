# Short code และ Prompt สำหรับสร้างโปรเจกต์

เอกสารประกอบ `presentation_flow_th.md` โดยคัดเฉพาะฟังก์ชันที่จำเป็นต่อการอธิบายระบบ และ prompt สำหรับสั่ง coding agent

> โค้ดเป็นตัวอย่างย่อเพื่อสื่อแนวคิด ไม่ใช่โค้ด production ที่แทนไฟล์จริงได้ทันที

## Short code สำหรับอธิบาย

### 1. Ingest: แบ่งเอกสารและเก็บ embedding

```python
def ingest_text_file(path, embedder, db):
    text = path.read_text(encoding="utf-8").strip()
    chunks = [part.strip() for part in text.split("\n\n") if part.strip()]

    for index, chunk in enumerate(chunks, start=1):
        db.add_text(
            text=chunk,
            embedding=embedder.embed_text(chunk),
            source_name=path.name,
            chunk_id=f"{path.name}#chunk-{index}",
        )
```

**อธิบาย:** แบ่งไฟล์ตามย่อหน้า แปลงข้อความเป็นเวกเตอร์ แล้วเก็บข้อความและ metadata ลง ChromaDB เพื่อค้นกลับไปยังแหล่งต้นทางได้

### 2. Retrieve: ค้นหลักฐานที่เกี่ยวข้อง

```python
def retrieve_evidence(question, top_k=3):
    query_vector = embedder.embed_query(question)
    results = db.search_all(
        query_embedding=query_vector,
        n_results=top_k,
    )
    return [
        item for item in results["texts"]
        if item["text"].strip() and item["retrieval_score"] <= 0.22
    ]
```

**อธิบาย:** แปลงคำถามเป็นเวกเตอร์และค้น chunk ใกล้ที่สุด คะแนนเป็น cosine distance ดังนั้นค่ายิ่งต่ำยิ่งใกล้ ตัวอย่างใช้ top-k 3 และ cutoff 0.22 ตามค่าเริ่มต้นของ backend ปัจจุบัน

### 3. Context: จัดหลักฐานและ citation ให้ LLM

```python
def make_context(evidence):
    if not evidence:
        return "ไม่พบข้อมูลที่เพียงพอจากแหล่งข้อมูล"

    blocks = []
    for index, item in enumerate(evidence, start=1):
        blocks.append(
            f"[S{index}] {item['source_name']} / {item['chunk_id']}\n{item['text']}"
        )

    return "\n---\n".join(blocks) + (
        "\n\nตอบจากหลักฐานนี้เท่านั้น อ้างอิงด้วย [S#] ห้ามเดา "
        "และหากหลักฐานไม่พอให้แจ้งว่าไม่พบข้อมูลที่เพียงพอ"
    )
```

**อธิบาย:** ใส่ citation ID ให้หลักฐานที่ค้นได้จริง และส่งกติกา grounding ไปพร้อม context เพื่อให้ตรวจย้อนกลับได้

### 4. Generate: ให้โมเดลสร้างคำตอบ

```python
def answer_with_rag(model, question, context):
    prompt = f"คำถาม: {question}\n\nหลักฐาน:\n{context}\n\nคำตอบ:"
    return ollama_chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
```

**อธิบาย:** RAG ส่งทั้งคำถามและ context ให้ Ollama ส่วน baseline ส่งคำถามอย่างเดียว โดยไม่มีขั้น retrieval และไม่มี context

### 5. Evaluation: วนชุดคำถามและเก็บผล

```python
for item in questions:
    if use_rag:
        evidence = retrieve_evidence(item["question"], top_k=3)
        context = make_context(evidence)
    else:
        context = ""

    answer = generate_answer(item["question"], context)
    save_result(item["id"], item["question"], answer, context)
```

**อธิบาย:** ใช้ dataset เดียวกันวนทีละข้อและเก็บคำตอบกับ context เพื่อเทียบกับ expected answer ภายหลัง การทดลองจริงใช้ runner แยกสำหรับ baseline และ RAG

## Prompt สั่งสร้างโปรเจกต์

```text
สร้างโปรเจกต์เว็บถามตอบเอกสารภาษาไทยเพื่อทดลองเปรียบเทียบ Baseline LLM กับ RAG แบบ end to end

เป้าหมาย
- ผู้ใช้พิมพ์คำถามและเลือกได้ว่าจะใช้ Baseline (ไม่ค้นเอกสาร) หรือ RAG (ค้นหลักฐานก่อนตอบ)
- แสดงคำตอบ แหล่งข้อมูล citation หลักฐานที่ค้นได้ และเวลาในการทำงาน
- มี evaluation runner สำหรับ question dataset 50 ข้อ และส่งออกผล JSON/CSV
- ใช้โมเดลในเครื่องผ่าน Ollama

เทคโนโลยี
- Frontend: HTML/CSS/JavaScript พร้อม Vite
- Backend: Python FastAPI
- LLM runtime: Ollama API
- Vector database: ChromaDB ใช้ cosine distance
- Embedding model: multilingual embedding ที่รองรับภาษาไทย

ความสามารถ
1. Ingestion อ่านไฟล์ txt, md, csv, json จากโฟลเดอร์ dataset แบ่งข้อความเป็น chunks ตามย่อหน้า สร้าง embedding และบันทึกข้อความพร้อม source_name, document_id และ chunk_id
2. RAG API รับ question, top_k และ options จากนั้น embed คำถาม ค้น chunk ที่เกี่ยวข้อง และคืนข้อความ คะแนน rank และ metadata ของแหล่งข้อมูล
3. Evidence gate กรอง chunk ว่างและหลักฐานที่ไม่ผ่าน distance threshold หากไม่มีหลักฐานเพียงพอให้ตอบว่าไม่พบข้อมูลที่เพียงพอจากแหล่งข้อมูล
4. Grounded generation ส่งคำถามและ context ให้ LLM โดยกำชับให้ใช้ context เป็นหลักฐานเท่านั้น ห้ามแต่งข้อเท็จจริง และให้อ้าง citation ID ที่มีอยู่จริง
5. Baseline ส่งเฉพาะคำถามให้ LLM โดยปิด retrieval/context และบันทึก rag_used=false
6. Frontend มีช่องถามตอบ ตัวเลือกโมเดล สวิตช์ RAG และส่วนแสดงคำตอบ citations กับหลักฐานที่ค้นได้
7. Evaluation โหลด dataset ที่มี id, question, expected_answer รันทีละข้อทั้งสองโหมด และเก็บ answer, context, sources, errors, token counts และ latency ใน JSON/CSV
8. Reproducibility บันทึก model, temperature, top_k, threshold, dataset path, timestamp และสถานะสำเร็จ/ผิดพลาดทุกครั้ง

ข้อกำหนดด้านความถูกต้อง
- ห้ามสร้างผลทดลอง คะแนน หรือ expected answer ขึ้นเอง
- เก็บ failed cases ในผลลัพธ์ ห้ามตัดออกเงียบ ๆ
- แยก retrieval quality, answer quality, faithfulness และ hallucination ออกจากกัน
- ห้ามกล่าวว่า RAG ลด hallucination หากยังไม่มีการประเมินคำตอบ
- หากต้องการวัดผลของ RAG ให้ใช้โมเดล dataset prompt และพารามิเตอร์เดียวกัน แล้วเปลี่ยนเฉพาะ RAG off/on
- citation ต้องเชื่อมกลับไปยัง source และ chunk ที่มีอยู่จริง
- ตรวจ input ว่าง จัดการ timeout/API errors และอ่าน URL/config จาก environment variables

รูปแบบงานที่ต้องส่ง
- ตรวจ repository ปัจจุบันก่อนแก้ และรักษาฟังก์ชันเดิมที่ยังใช้งานได้
- แยกโค้ดเป็น frontend, backend, scripts, dataset และ results อย่างอ่านง่าย
- เพิ่ม README ขั้นตอนติดตั้ง เปิดบริการ ingest เอกสาร และรัน evaluation
- เพิ่ม requirements และ package scripts ที่จำเป็น
- เขียนโค้ดที่รันได้จริง ไม่ใช่ pseudocode
- เมื่อเสร็จ สรุป architecture, flow, วิธีรัน, ไฟล์สำคัญ และข้อจำกัดของผลที่วัดได้

เริ่มจากตรวจ repository และบอกแผนไฟล์ที่จะสร้างหรือแก้ จากนั้นลงมือสร้างโปรเจกต์ให้ครบตามข้อกำหนด
```
