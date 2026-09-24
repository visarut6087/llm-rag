# Minimal RAG Improvement — Reduce Hallucination & Improve Efficiency

## Objective

ปรับปรุงระบบ Thai Local LLM + RAG ที่มีอยู่แล้ว โดยเน้น:
1. ลด Hallucination
2. เพิ่มความถูกต้องจาก Evidence
3. ปรับ Abstention ให้ดีขึ้น
4. ลด unnecessary processing / latency
5. เปลี่ยน code ให้น้อยที่สุด
6. รักษา architecture เดิม
7. ห้ามสร้างข้อมูลหรือผลการทดลองปลอม

## Current System

- Frontend: Vite
- Backend: FastAPI
- Vector DB: ChromaDB
- Text embedding: `intfloat/multilingual-e5-large`
- Image embedding: `clip-ViT-B-32`
- LLM: Ollama `qwen2.5:7b`
- Dataset: `backend/dataset/sarabun 26.txt`
- Dataset size: 44 text chunks

## Current Evidence

### LLM evaluation

```text
baseline_rag:
Faithfulness           = 83.3%
Answer relevance       = 100.0%
Context relevance      = 100.0%
Citation correctness   = 100.0%
Hallucination rate     = 16.7%
Abstention correctness = 66.7%

reranking:
Faithfulness           = 50.0%
Answer relevance       = 83.3%
Context relevance      = 83.3%
Citation correctness   = 83.3%
Hallucination rate     = 33.3%
Abstention correctness = 50.0%

conversation_memory:
Faithfulness           = 50.0%
Answer relevance       = 83.3%
Context relevance      = 83.3%
Citation correctness   = 83.3%
Hallucination rate     = 33.3%
Abstention correctness = 50.0%
```

### Retrieval evaluation

```text
baseline_rag:
Recall@3    = 1.000
Precision@3 = 0.333
MRR         = 1.000
nDCG@3      = 1.000

reranking:
Recall@3    = 0.400
Precision@3 = 0.133
MRR         = 0.133
nDCG@3      = 0.200

conversation_memory:
Recall@3    = 0.400
Precision@3 = 0.133
MRR         = 0.133
nDCG@3      = 0.200
```

Interpretation:
- Baseline retrieval is currently strong on the current evaluation set.
- Reranking currently hurts retrieval quality.
- Conversation memory currently hurts retrieval quality.
- Do not assume either feature improves the system.
- Do not force them into the final default pipeline.

# Required Changes

## 1. Grounded Answering

Modify the existing answer-generation behavior so the LLM:
- Uses retrieved context as the factual source.
- Does not add unsupported facts.
- Does not rely on general knowledge when RAG context is insufficient.
- Clearly states when retrieved information is insufficient.
- Avoids guessing.
- Keeps answers concise when evidence is limited.

Prefer modifying the existing system prompt or answer construction.
Do not introduce another LLM call unless absolutely necessary.

## 2. Lightweight Evidence Sufficiency / Abstention

Add a lightweight evidence sufficiency check before final answer generation.

Reuse existing retrieval information whenever possible:
- top retrieval score / distance
- number of retrieved chunks
- existing relevance score
- whether retrieved text contains usable evidence

Do not invent an arbitrary threshold without testing.

If a new threshold is required:
1. Make it configurable.
2. Use a conservative initial value.
3. Document it.
4. Evaluate it.
5. Do not tune it directly against the final test set.

Desired flow:

```text
Question
   ↓
Retrieve
   ↓
Evidence sufficient?
   ├── YES → grounded answer + citation
   └── NO  → abstain / say insufficient evidence
```

Abstention must be natural Thai and must not fabricate an answer.

## 3. Keep Baseline Retrieval

Because baseline currently performs better:
- Keep baseline vector retrieval as the default.
- Do not enable reranking by default.
- Do not use conversation memory as factual evidence.
- Keep existing feature switches for future experiments.
- Do not delete existing reranker or memory code.

## 4. Conservative Query Rewriting

Keep query rewriting available but use it only when needed:
- follow-up questions
- pronoun/coreference
- incomplete conversational questions
- ambiguous references to previous context

For clear standalone Thai questions:

```text
Clear standalone query?
   ├── YES → use original query
   └── NO  → rewrite query
```

Preserve the original question for logging/evaluation.
Avoid unnecessary LLM/API calls.

## 5. Reranking

Do not redesign the reranker in this phase.

Current evidence:
```text
Baseline Recall@3 = 1.000
Reranking Recall@3 = 0.400
```

Inspect only enough to detect an obvious integration or implementation problem.

If there is a concrete bug, fix it.
If there is no obvious bug, leave the reranker unchanged and disabled by default.

Do not optimize the reranker against the current evaluation set.

## 6. Conversation Memory

Separate:
```text
Conversation Memory
= intent / reference / conversational context

Retrieved Documents
= factual evidence
```

Memory may help resolve follow-up questions, but factual claims must still be supported by retrieved documents.

For clear standalone questions, avoid unnecessary memory processing.
Keep the feature switch for future experiments.

## 7. Citation

Preserve the existing citation implementation because current citation correctness is 100%.

Do not redesign citation format unless there is a concrete bug.

Citations must point only to actual retrieved sources/chunks.

Never fabricate:
- source
- document ID
- chunk ID
- citation
- quotation
- retrieval score

# Efficiency

Make only low-risk optimizations.

Prefer:
- avoid duplicate embedding calls
- avoid duplicate retrieval calls
- avoid duplicate query rewriting
- avoid unnecessary LLM calls
- reuse existing embeddings
- reuse retrieved context
- keep Top-K reasonable
- avoid recomputing values within one request

Do NOT:
- replace the embedding model
- replace ChromaDB
- replace Ollama
- replace the LLM
- redesign the frontend
- rewrite the backend architecture
- introduce a new vector database
- add a second LLM
- add a heavy reranker
- add complex agent behavior

# Files To Inspect First

Inspect only:
```text
backend/main.py
backend/db.py
backend/embedder.py
backend/query_rewriter.py
backend/reranker.py
backend/ingest_dataset.py
scripts/evaluate_rag.py
scripts/evaluate_llm.py
scripts/run_ablation.py
docs/system_audit.md
docs/integrated_system.md
docs/rag_evaluation.md
docs/llm_evaluation.md
```

Do not scan or rewrite the entire repository unnecessarily.

# Implementation Rules

1. Preserve existing API compatibility.
2. Preserve existing frontend behavior.
3. Preserve existing dataset.
4. Preserve existing ChromaDB schema unless necessary.
5. Preserve existing embeddings.
6. Preserve existing Ollama model.
7. Keep new behavior configurable where practical.
8. Use the smallest reasonable code change.
9. Do not fabricate evaluation results.
10. Do not modify unrelated features.

# Verification

Before evaluation verify:
```text
Backend HTTP 200
RAG stats available
Dataset = 44 chunks
Ollama model available
Frontend build works
```

Run:
```bash
npm run build
```

# Evaluation

## Retrieval

```bash
./.venv/bin/python scripts/evaluate_rag.py \
  --base-url http://127.0.0.1:8000 \
  --output-dir results/evaluation/improved_rag
```

## LLM

```bash
./.venv/bin/python scripts/evaluate_llm.py \
  --rag-url http://127.0.0.1:8000 \
  --ollama-url http://127.0.0.1:11434 \
  --model qwen2.5:7b \
  --output-dir results/evaluation/improved_llm
```

# Compare Against Baseline

Create:
```text
                    Baseline    Improved
Faithfulness
Answer relevance
Context relevance
Citation correctness
Hallucination rate
Abstention correctness
Latency
```

Also compare retrieval metrics where applicable.

Do not claim improvement unless supported by actual results.
If a metric becomes worse, report it clearly.

# Research Integrity

This is a research project.

Do NOT:
- fabricate scores
- fabricate evaluation cases
- remove failed cases
- cherry-pick successful cases
- tune directly against final test results
- claim "reduced hallucination" without baseline comparison
- claim "improved accuracy" without evidence
- claim "faster" without latency measurement

A negative result is valid and must be reported.

# Expected Output

After implementation, report only:
1. What was changed
2. Files changed
3. Why each change was necessary
4. Retrieval results before/after
5. LLM results before/after
6. Latency before/after if measured
7. Remaining weaknesses
8. Recommended next step

Keep the report concise.

# Stop Rule

IMPORTANT:
Do NOT continue to another research phase automatically.

After:
- implementation
- basic verification
- evaluation
- result comparison

STOP and wait for the next instruction.

Do not implement a new reranker.
Do not redesign conversation memory.
Do not add new evaluation methodology.
Do not modify the dataset.

Goal:
> Make the smallest practical changes that reduce unsupported answers and unnecessary processing while preserving the currently strong baseline retrieval performance.
