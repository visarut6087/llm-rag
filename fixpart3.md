# RAG Evaluation Dataset Expansion Plan

## Objective

เพิ่มจำนวน evaluation cases ของระบบ Thai Local LLM + RAG ก่อนปรับปรุงระบบลด Hallucination

เหตุผลหลัก:
- ปัจจุบันมีเพียง 6 cases
- ผลลัพธ์จึงเปลี่ยนแปลงมากเมื่อผิดหรือถูกเพียง 1 case
- ต้องการ baseline ที่น่าเชื่อถือขึ้นก่อนปรับระบบ
- ต้องการวัดทั้งการตอบจากหลักฐานและความสามารถในการ abstain เมื่อข้อมูลไม่เพียงพอ

## Current Baseline

จำนวน cases ปัจจุบัน: **6**

```text
Faithfulness           = 83.3%
Answer relevance       = 100.0%
Context relevance      = 100.0%
Citation correctness   = 100.0%
Hallucination rate     = 16.7%
Abstention correctness = 66.7%
```

โดยประมาณ:

```text
Faithfulness           = 5/6
Answer relevance       = 6/6
Context relevance      = 6/6
Citation correctness   = 6/6
Hallucination          = 1/6
Abstention correctness = 4/6
```

## Target Evaluation Size

เพิ่มเป็น **30 cases** เป็นเป้าหมายขั้นต่ำสำหรับรอบถัดไป

ไม่จำเป็นต้องทำให้ครบ 30 cases ด้วยการสุ่มคำถามซ้ำ ๆ ต้องเน้นความหลากหลายและความสามารถในการวัด Hallucination

## Recommended Case Distribution

| ประเภท | จำนวน | จุดประสงค์ |
|---|---:|---|
| Answerable: คำตอบตรงในบทความ | 10 | วัด factual grounding |
| Answerable: ต้องเชื่อมข้อมูลหลายส่วน | 5 | วัดการใช้หลาย chunks |
| Answerable: ข้อมูลเฉพาะ | 5 | วัด retrieval + generation |
| Follow-up / conversational | 3 | วัด conversation handling |
| Ambiguous | 2 | วัด query rewriting |
| Unanswerable: ไม่มีข้อมูลใน dataset | 5 | วัด abstention และ hallucination |
| **รวม** | **30** | |

## Important Research Rule

ห้ามสร้างคำถามโดยดูเฉพาะผลที่ระบบทำได้ดีหรือแย่

Evaluation set ต้องครอบคลุม:
- คำถามที่ตอบได้
- คำถามที่ต้องใช้หลายหลักฐาน
- คำถามสนทนาต่อเนื่อง
- คำถามกำกวม
- คำถามที่ไม่มีข้อมูลรองรับ

สำหรับ unanswerable cases ต้องตรวจสอบด้วย dataset จริงว่าไม่มีหลักฐานเพียงพอ

ห้ามกำหนดคำตอบที่ไม่มีอยู่ใน dataset เป็น ground truth

## Dataset / Ground Truth

ใช้ข้อมูลจาก:

```text
backend/dataset/sarabun 26.txt
```

ปัจจุบัน dataset มี 44 text chunks

สำหรับแต่ละ case ควรเก็บข้อมูลอย่างน้อย:

```text
id
question
expected_answer
answerable
relevant_source
relevant_chunk
question_type
```

สำหรับ unanswerable case:

```text
answerable = false
relevant_source = null
relevant_chunk = null
```

และ expected behavior คือระบบควร abstain แทนการเดา

## Evaluation Split

ถ้าเป็นไปได้ ให้แบ่ง cases เป็น:

```text
Development / Validation
    ↓
ใช้ตรวจและปรับระบบ

Final Test
    ↓
ห้ามใช้ผลเพื่อปรับระบบ
```

ตัวอย่าง:

```text
30 total cases

20 development
10 final test
```

หากยังมีข้อมูลไม่เพียงพอสำหรับ split นี้ ให้สร้าง 30 cases ก่อน และบันทึกไว้อย่างชัดเจนว่าเป็น evaluation set เดียวกัน

## Baseline Requirement

ก่อนปรับปรุงระบบ:

1. รัน baseline ด้วย evaluation set ใหม่
2. เก็บผลจริง
3. ห้ามปรับ prompt / threshold / algorithm ตามผล final test
4. บันทึกผลเป็น baseline ใหม่

จากนั้นจึงค่อยทำ:

```text
Baseline
   ↓
Minimal Grounding + Abstention Improvement
   ↓
Run same evaluation cases
   ↓
Compare
```

## Metrics

อย่างน้อยให้เก็บ:

```text
Faithfulness
Answer Relevance
Context Relevance
Citation Correctness
Hallucination Rate
Abstention Correctness
Latency
```

และสำหรับ retrieval:

```text
Recall@K
Precision@K
MRR
nDCG@K
```

## Interpretation

ห้ามสรุปว่า:

> "Hallucination ของภาษาไทยโดยทั่วไป = X%"

เพราะ evaluation set นี้เป็นของโปรเจกต์และมีขนาดจำกัด

ให้รายงานเป็น:

> "The proposed system achieved a hallucination rate of X% on the evaluation dataset."

เมื่อเปรียบเทียบระบบ ให้ใช้ชุด cases เดียวกัน:

```text
Baseline hallucination
        ↓
Improved hallucination
```

จึงจะสามารถรายงานการเปลี่ยนแปลงได้อย่างเหมาะสม

## Quality Control

ก่อนนำ case เข้า evaluation:

- อ่านบทความต้นทางจริง
- ตรวจว่า expected answer รองรับโดย dataset
- ระบุ relevant chunk
- ตรวจว่า unanswerable case ไม่มี evidence เพียงพอ
- ห้ามมีคำตอบที่อาศัยความรู้ภายนอกโดยไม่ระบุ
- ห้ามสร้าง citation ที่ไม่มีอยู่จริง
- ห้าม duplicate cases โดยไม่ตั้งใจ

## Codex Instructions

เมื่อทำงานนี้:

1. อ่าน evaluation scripts ที่มีอยู่ก่อน
2. ตรวจรูปแบบ evaluation case ปัจจุบัน
3. เพิ่ม cases โดยรักษา schema เดิมถ้าเป็นไปได้
4. ใช้ dataset จริงเท่านั้น
5. ห้ามแก้ RAG algorithm
6. ห้ามแก้ prompt ระบบ
7. ห้ามแก้ reranker
8. ห้ามแก้ conversation memory
9. ห้ามเพิ่ม feature ใหม่
10. ห้ามสร้างผล evaluation ปลอม
11. รัน validation เพื่อยืนยันว่า evaluation dataset โหลดได้
12. หากทำได้ ให้ตรวจ duplicate และ malformed cases
13. สรุปจำนวน cases และประเภทของ cases
14. STOP หลังสร้างและตรวจ evaluation set

## Efficiency Rules

เพื่อประหยัด token / compute:

- Inspect เฉพาะ evaluation scripts และไฟล์ dataset ที่จำเป็น
- ไม่ต้องอ่าน repository ทั้งหมด
- ไม่ต้องรัน LLM evaluation จนกว่าจะสร้าง cases เสร็จ
- ไม่ต้องรัน ablation
- ไม่ต้องแก้ implementation
- ไม่ต้องสร้าง embedding ใหม่ถ้าไม่จำเป็น
- ใช้ existing evaluation infrastructure

## Expected Output

รายงานสั้น ๆ:

```text
Total cases:
Answerable:
Multi-evidence:
Follow-up:
Ambiguous:
Unanswerable:

Files changed:
Validation result:
Duplicate check:
Next step:
```

## Stop Rule

หลังจาก:

- สร้าง evaluation cases
- ตรวจ ground truth
- ตรวจ schema
- ตรวจว่า evaluation script โหลด cases ได้

ให้ STOP

อย่าเพิ่งปรับ RAG
อย่าเพิ่งปรับ LLM
อย่าเพิ่งปรับ reranker
อย่าเพิ่งปรับ memory
อย่าเพิ่งทำ ablation

ขั้นตอนถัดไปจะเป็นการรัน **Baseline evaluation บนชุด 30 cases** ก่อน
