---
name: rag-research-roadmap
description: Phased research roadmap for developing and evaluating an LLM-RAG system to reduce hallucinations in Thai article knowledge.
argument-hint: "[phase]"
agent: agent
---

# LLM-RAG Thai Hallucination Research

## 1. Research Objective

Develop and experimentally evaluate an LLM-RAG system for:

**Reducing hallucinations when answering questions about Thai-language articles.**

The project must be treated as a research project, not merely a software feature implementation.

Primary goals:

1. Improve retrieval relevance.
2. Improve factual grounding.
3. Make generated answers traceable to source evidence.
4. Handle conversational follow-up questions.
5. Reduce hallucinated claims.
6. Measure each component independently.
7. Determine which components actually improve the system.

Do not assume that every added component improves performance.

---

# 2. Current System

The existing system contains the core components described in `README.md`:

- Vite frontend
- FastAPI backend
- Ollama
- ChromaDB
- text embeddings
- image embeddings
- RAG query
- optional web search
- chat history

Preserve the existing working architecture.

Do not rewrite the entire application.

Inspect the current implementation before modifying anything.

---

# 3. Research Principle

Every new component must be evaluated against the previous version.

Use:

```text
Baseline
↓
Baseline + Component 1
↓
Baseline + Component 1 + Component 2
↓
...
```

The purpose is to determine the contribution of each component.

Do not enable all proposed features simultaneously before their individual behavior has been verified.

---

# 4. Phase Order

Implement the project in this order:

```text
Phase 0
Current System Audit

Phase 1
RAG Relevance Score

Phase 2
RAG Citations / Sources

Phase 3
Reranking

Phase 4
Query Rewriting

Phase 5
Conversation Memory → RAG

Phase 6
RAG Evaluation

Phase 7
LLM Evaluation

Phase 8
Integrated System

Phase 9
Ablation and Final Research Analysis
```

---

# 5. Phase Control

IMPORTANT:

Do NOT execute all phases automatically.

At the beginning of every interaction:

1. Read this file.
2. Inspect the current project state.
3. Determine which phase is requested.
4. Work ONLY on that phase.
5. Verify the implementation.
6. Report the result.
7. STOP.

Wait for the user to explicitly request the next phase.

Do not continue automatically.

---

# 6. Phase 0 — Current System Audit

Before implementing research features, inspect:

```text
src/
backend/
README.md
package.json
backend/requirements.txt
```

Determine:

- current frontend flow
- current API flow
- current RAG flow
- embedding model
- ChromaDB structure
- chunking strategy
- retrieval implementation
- Ollama request flow
- chat history implementation
- web search implementation
- existing metadata
- existing evaluation capability

Create:

```text
docs/system_audit.md
```

Classify every component:

```text
IMPLEMENTED
PARTIAL
MISSING
UNKNOWN
```

Do not modify application behavior during the audit unless required to run the audit.

### Phase 0 completion

The audit must be complete before Phase 1 begins.

STOP after Phase 0.

---

# 7. Phase 1 — RAG Relevance Score

## Objective

Make retrieval quality measurable.

For every retrieved document/chunk, expose at least:

```text
document_id
source
chunk_id
retrieval_score
rank
text
```

If ChromaDB already returns a native distance/score, preserve its semantics.

Do not rename distance as similarity without verification.

If conversion is needed:

```text
distance → similarity
```

document the formula and direction clearly.

---

## Required behavior

For every RAG query:

```text
User Query
    ↓
Embedding
    ↓
ChromaDB Retrieval
    ↓
Top-K Results
    ↓
Score + Rank
```

The system must be able to answer:

> Why was this document retrieved?

---

## Required UI/API information

Return enough metadata to inspect:

```text
Rank
Source
Chunk
Score
```

Do not expose unnecessary internal information.

---

## Research output

Create:

```text
results/retrieval/
```

Store retrieval examples and measurements there.

---

## Phase 1 completion

Verify:

- retrieval score exists
- score semantics are documented
- ranking is correct
- source metadata is preserved
- existing RAG still works

STOP.

---

# 8. Phase 2 — RAG Citations / Sources

## Objective

Make generated answers traceable to retrieved evidence.

The system should identify which retrieved source supports the generated answer.

Desired flow:

```text
Question
 ↓
Retrieve
 ↓
Ranked Evidence
 ↓
LLM
 ↓
Answer + Sources
```

---

## Citation design

Every retrieved item should have stable metadata:

```text
source_id
source_name
document_id
chunk_id
```

The generated response should be able to reference these sources.

Example:

```text
คำตอบ ...

[แหล่งข้อมูล 1]
บทความ: ...
ส่วนที่ใช้: ...
```

Do not fabricate citations.

If the answer cannot be supported by retrieved evidence:

```text
ไม่พบข้อมูลที่เพียงพอจากแหล่งข้อมูล
```

or another clearly defined abstention response.

---

## Grounding rule

Conversation history must not automatically become factual evidence.

Retrieved source content is the factual grounding source.

This separation is important for hallucination reduction in conversational RAG; recent multi-turn RAG work explicitly separates conversation history used for intent/coreference from retrieved passages used for factual grounding.

---

## Phase 2 completion

Verify:

- every source has stable identity
- answer can display sources
- source references correspond to actual retrieved documents
- unsupported claims can be identified
- no fabricated citation

STOP.

---

# 9. Phase 3 — Reranking

## Objective

Improve the ordering of retrieved candidates before sending context to the LLM.

Flow:

```text
Query
 ↓
Initial Retrieval
 ↓
Top-K Candidates
 ↓
Reranker
 ↓
Re-ranked Candidates
 ↓
Top-N Context
 ↓
LLM
```

Do not replace the existing retriever immediately.

First implement:

```text
Retriever → Reranker
```

so the effect can be measured independently.

---

## Reranking requirements

Record:

```text
initial_rank
initial_score
rerank_score
final_rank
source
chunk_id
```

The reranker must not silently discard source metadata.

---

## Research comparison

Compare:

```text
Retrieval only
vs
Retrieval + Reranking
```

Measure retrieval quality and downstream answer quality.

Recent RAG research commonly uses retrieval followed by cross-encoder/LLM reranking, including recent multi-turn RAG systems.

---

## Phase 3 completion

Verify:

- reranker works
- ranking changes are observable
- metadata survives reranking
- latency is measured
- retrieval-only baseline remains available

STOP.

---

# 10. Phase 4 — Query Rewriting

## Objective

Improve retrieval for ambiguous, incomplete, or conversational Thai questions.

Example:

```text
User:
แล้วเรื่องนี้เกิดขึ้นเมื่อไหร่?

Conversation:
ก่อนหน้านี้พูดถึงเหตุการณ์ X
```

Rewrite into a standalone retrieval query:

```text
เหตุการณ์ X เกิดขึ้นเมื่อไหร่?
```

---

## Critical rule

Maintain two separate representations:

```text
Original user question
        ↓
Query rewriting
        ↓
Retrieval query
```

Do NOT replace the original question.

The original question remains the user-visible question.

---

## Compare

```text
Original Query
vs
Rewritten Query
```

Measure whether rewriting improves retrieval.

Query rewriting can help conversational retrieval, but recent research also shows that rewriting effects vary by retriever and can introduce or mask retrieval biases. Therefore it must be evaluated experimentally rather than assumed beneficial.

---

## Phase 4 completion

Verify:

- original question preserved
- rewritten query logged
- retrieval uses intended query
- rewrite failures do not break RAG
- latency measured

STOP.

---

# 11. Phase 5 — Conversation Memory → RAG

## Objective

Allow previous conversation turns to improve retrieval without allowing conversation history to become unsupported factual evidence.

Flow:

```text
Conversation History
        │
        ▼
Intent / Coreference
        │
        ▼
Query Rewriting
        │
        ▼
Retrieval
        │
        ▼
Evidence
        │
        ▼
LLM
```

Important separation:

```text
Conversation history
= context for understanding the question

Retrieved documents
= factual evidence
```

Do not treat previous assistant responses as authoritative evidence.

---

## Test cases

At minimum test:

### Case A
Standalone question

### Case B
Follow-up question

### Case C
Pronoun/coreference

### Case D
Topic change

### Case E
Ambiguous follow-up

### Case F
Question whose answer is not present in the knowledge base

---

## Phase 5 completion

Verify:

- multi-turn retrieval works
- query rewriting uses relevant history
- irrelevant history does not contaminate retrieval
- unsupported information does not become evidence
- source citations still work

STOP.

---

# 12. Phase 6 — RAG Evaluation

## Objective

Evaluate retrieval independently from generation.

Create a small research evaluation dataset based on Thai articles.

Each item should contain:

```text
question
expected_answer
relevant_source
relevant_chunk
answerable
```

Do not fabricate evaluation labels.

If manual annotation is required, store the annotation explicitly.

---

## Retrieval metrics

Where applicable measure:

```text
Recall@K
Precision@K
Hit Rate@K
MRR
nDCG@K
```

Also measure:

```text
retrieval latency
reranking latency
query rewriting latency
```

---

## Research comparison

Compare:

```text
Baseline RAG
+ Relevance Score
+ Reranking
+ Query Rewriting
+ Conversation Memory
```

Do not combine all improvements without preserving individual baselines.

---

## Phase 6 completion

Generate:

```text
results/evaluation/rag_evaluation.json
results/evaluation/rag_evaluation.csv
```

and:

```text
docs/rag_evaluation.md
```

STOP.

---

# 13. Phase 7 — LLM Evaluation

## Objective

Determine whether improved retrieval actually reduces hallucination.

Retrieval improvement alone is insufficient.

Evaluate generated answers on:

### Faithfulness

Is the answer supported by retrieved evidence?

### Answer relevance

Does the answer actually address the question?

### Context relevance

Is the retrieved context relevant to the question?

### Citation correctness

Do citations actually support the associated claims?

### Hallucination rate

How often does the answer contain unsupported factual claims?

### Abstention correctness

When evidence is insufficient, does the system avoid inventing an answer?

---

## Important distinction

Evaluate at least:

```text
Retrieval quality
        ≠
Answer quality
        ≠
Faithfulness
        ≠
Hallucination rate
```

Do not use one metric as a substitute for all four.

---

## LLM evaluation dataset

Include:

```text
answerable questions
unanswerable questions
ambiguous questions
multi-turn questions
questions requiring specific source evidence
```

The unanswerable set is important because a hallucination-reduction system should know when the knowledge base does not contain sufficient evidence.

Recent multi-turn RAG evaluations explicitly include unanswerable queries and evaluate retrieval and generation separately.

---

## Phase 7 completion

Generate:

```text
results/evaluation/llm_evaluation.json
results/evaluation/llm_evaluation.csv
docs/llm_evaluation.md
```

STOP.

---

# 14. Phase 8 — Integrated System

Only after Phases 1–7 have been individually verified, combine:

```text
User
 ↓
Conversation Memory
 ↓
Query Rewriting
 ↓
Retrieval
 ↓
Relevance Score
 ↓
Reranking
 ↓
Top-N Evidence
 ↓
Context Construction
 ↓
LLM
 ↓
Grounded Answer
 ↓
Citation
```

The final system should have an explicit grounding rule:

```text
Use retrieved evidence as the factual basis.
If evidence is insufficient, say so.
Do not invent unsupported facts.
```

---

# 15. Phase 9 — Ablation

Perform controlled comparisons:

```text
A Baseline RAG

B + Relevance Score

C + Citations

D + Reranking

E + Query Rewriting

F + Conversation Memory

G Full RAG
```

For each configuration measure:

```text
Retrieval Quality
Answer Relevance
Faithfulness
Hallucination Rate
Citation Accuracy
Latency
```

The purpose is to identify which component actually contributes to hallucination reduction.

---

# 16. Final Research Questions

The completed project should answer:

### RQ1
Does improved retrieval relevance reduce hallucinations?

### RQ2
Does reranking improve the quality of evidence supplied to the LLM?

### RQ3
Does query rewriting improve retrieval for Thai conversational questions?

### RQ4
Does conversation-aware retrieval improve multi-turn question answering without introducing unsupported facts?

### RQ5
Does citation-grounded generation improve factual traceability?

### RQ6
Which component contributes most to hallucination reduction?

### RQ7
What is the trade-off between hallucination reduction, answer quality, and latency?

---

# 17. Scientific Integrity

Never:

- fabricate evaluation labels
- fabricate retrieval scores
- fabricate citations
- fabricate hallucination rates
- claim hallucination reduction without a baseline
- tune on the final test set
- remove failed cases
- cherry-pick favorable examples
- claim a component helps before evaluating it

Negative results must be preserved.

---

# 18. Token / Development Efficiency

For every phase:

- inspect only relevant files
- reuse existing code
- avoid repository-wide rewrites
- run small tests before expensive evaluation
- do not run future phases automatically
- do not repeat completed experiments
- keep final response concise

---

# 19. Phase Gate

A phase is complete only when:

```text
[ ] Implementation complete
[ ] Relevant tests pass
[ ] Actual experiment executed where required
[ ] Results saved
[ ] No fabricated values
[ ] Existing functionality still works
[ ] Documentation updated
[ ] Phase conclusion recorded
```

Then STOP.

Wait for the user to request the next phase.

---

# 20. First Command

When this prompt is first invoked:

DO NOT implement anything.

Perform only:

```text
Phase 0 — Current System Audit
```

Return:

1. Current architecture
2. Existing capabilities
3. Missing capabilities
4. Recommended Phase 1
5. Files likely to change

Then STOP.

The user will explicitly authorize the next phase.