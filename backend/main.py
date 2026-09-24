import os

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from embedder import get_embedder
from db import get_db
from ingest_dataset import process_dataset_folder
from reranker import rerank_results
from query_rewriter import rewrite_query


DEFAULT_EVIDENCE_MAX_DISTANCE = 0.22


def get_evidence_max_distance() -> float:
    """Return the configurable cosine-distance cutoff for usable evidence."""
    try:
        return float(os.getenv("RAG_EVIDENCE_MAX_DISTANCE", DEFAULT_EVIDENCE_MAX_DISTANCE))
    except (TypeError, ValueError):
        return DEFAULT_EVIDENCE_MAX_DISTANCE


def is_usable_text_evidence(result: dict, max_distance: float) -> bool:
    text = (result.get("text") or result.get("content") or "").strip()
    distance = result.get("retrieval_score")
    return bool(text) and isinstance(distance, (int, float)) and distance <= max_distance

app = FastAPI(title="Multimodal RAG API Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextIngestRequest(BaseModel):
    text: str
    source_name: Optional[str] = "manual_input"

class RAGQueryRequest(BaseModel):
    prompt: str
    n_results: Optional[int] = 3
    use_reranker: bool = False
    use_query_rewriting: bool = True
    use_citations: bool = True
    # Explicit rewrite-only context. It is not used as factual evidence.
    conversation_context: Optional[list[dict]] = None

@app.on_event("startup")
def startup_event():
    print("Pre-loading Multimodal Embedder (ViT / CLIP)...")
    get_embedder()
    get_db()
    print("Multimodal RAG Server is ready!")

@app.get("/")
def root():
    return {"message": "Multimodal RAG Server is running", "status": "ok"}

@app.get("/api/rag/stats")
def get_stats():
    db = get_db()
    return db.get_stats()

@app.post("/api/ingest/text")
def ingest_text(req: TextIngestRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    source_name = req.source_name or "manual_input"
    
    embedder = get_embedder()
    db = get_db()
    
    # Split text into chunks (naive splitting by double newline / paragraph)
    paragraphs = [p.strip() for p in req.text.split("\n\n") if p.strip()]
    added_ids = []
    
    for index, p in enumerate(paragraphs, start=1):
        vec = embedder.embed_text(p)
        doc_id = db.add_text(
            text=p,
            embedding=vec,
            source_name=source_name,
            chunk_id=f"{source_name}#chunk-{index}",
        )
        added_ids.append(doc_id)
        
    return {
        "status": "success",
        "chunks_added": len(added_ids),
        "source": source_name
    }

@app.post("/api/ingest/image")
async def ingest_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    contents = await file.read()
    embedder = get_embedder()
    db = get_db()
    
    vec = embedder.embed_image(contents)
    doc_id = db.add_image(image_bytes=contents, embedding=vec, image_name=file.filename)
    
    return {
        "status": "success",
        "doc_id": doc_id,
        "image_name": file.filename
    }

@app.post("/api/ingest/dataset")
def ingest_dataset_endpoint(folder_path: Optional[str] = None):
    return process_dataset_folder(folder_path)

@app.post("/api/rag/query")
def query_rag(req: RAGQueryRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
    import time

    rewrite_started = time.perf_counter()
    rewrite_method = "deterministic_contextual_follow_up_v1"
    if req.use_query_rewriting:
        try:
            rewritten_query, rewrite_used_context = rewrite_query(
                req.prompt,
                req.conversation_context,
            )
        except Exception:
            # Query rewriting is an optimization; retrieval must remain available
            # if malformed optional context or a future rewriter fails.
            rewritten_query = req.prompt
            rewrite_used_context = False
            rewrite_method = "fallback_original_query"
    else:
        rewritten_query = req.prompt
        rewrite_used_context = False
        rewrite_method = "disabled_original_query"
    rewrite_latency_ms = round((time.perf_counter() - rewrite_started) * 1000, 3)

    embedder = get_embedder()
    db = get_db()
    
    # 1. Embed the rewritten query for retrieval. The original prompt remains
    # the user-visible question and is not replaced in the chat history.
    query_vec = embedder.embed_query(rewritten_query)
    
    # 2. Also encode with CLIP for cross-modal image search (512 dims)
    image_query_vec = embedder.embed_image_query(rewritten_query)
    
    final_k = max(req.n_results or 3, 1)
    candidate_k = final_k * 3 if req.use_reranker else final_k

    # 3. Initial ChromaDB retrieval. Reranking is a separate second stage.
    retrieval_started = time.perf_counter()
    results = db.search_all(
        query_embedding=query_vec,
        image_query_embedding=image_query_vec,
        n_results=candidate_k,
    )
    retrieval_latency_ms = round((time.perf_counter() - retrieval_started) * 1000, 3)

    reranking_started = time.perf_counter()
    if req.use_reranker:
        results["texts"] = rerank_results(rewritten_query, results.get("texts", []), text_key="text")[:final_k]
        results["images"] = rerank_results(rewritten_query, results.get("images", []), text_key="name")[:final_k]
    else:
        for result_type in ("texts", "images"):
            for index, result in enumerate(results.get(result_type, []), start=1):
                result["initial_rank"] = result.get("rank", index)
                result["initial_score"] = result.get("retrieval_score")
                result["rerank_score"] = None
                result["final_rank"] = result.get("rank", index)
    reranking_latency_ms = round((time.perf_counter() - reranking_started) * 1000, 3)
    
    # 3. Assign query-local citation labels to the actual retrieved records.
    # These labels are only valid for this query; stable source identity remains
    # in source_id/source_name/document_id/chunk_id.
    retrieved_sources = []
    for result_type, result_list in (("text", results.get("texts", [])), ("image", results.get("images", []))):
        for result in result_list:
            citation_id = f"S{len(retrieved_sources) + 1}"
            result["citation_id"] = citation_id
            retrieved_sources.append({
                "citation_id": citation_id,
                "type": result_type,
                "source_id": result.get("source_id"),
                "source_name": result.get("source_name") or result.get("source"),
                "document_id": result.get("document_id"),
                "chunk_id": result.get("chunk_id"),
                "retrieval_score": result.get("retrieval_score"),
                "rank": result.get("rank"),
                "initial_rank": result.get("initial_rank"),
                "initial_score": result.get("initial_score"),
                "rerank_score": result.get("rerank_score"),
                "final_rank": result.get("final_rank"),
            })

    text_results = results.get("texts", [])
    evidence_max_distance = get_evidence_max_distance()
    evidence_results = [
        result for result in text_results
        if is_usable_text_evidence(result, evidence_max_distance)
    ]
    evidence_sufficient = bool(evidence_results)
    evidence_reason = (
        "usable_text_evidence"
        if evidence_sufficient
        else "no_usable_text_evidence_within_distance_cutoff"
    )
    if req.use_citations:
        text_contexts = [
            "[{citation_id}] source_id={source_id} source_name={source_name} "
            "document_id={document_id} chunk_id={chunk_id}\n{text}".format(
                citation_id=result["citation_id"],
                source_id=result.get("source_id"),
                source_name=result.get("source_name") or result.get("source"),
                document_id=result.get("document_id"),
                chunk_id=result.get("chunk_id"),
                text=result.get("text") or result.get("content", ""),
            )
            for result in evidence_results
        ]
    else:
        text_contexts = [result.get("text") or result.get("content", "") for result in evidence_results]
    images_found = results.get("images", [])
    
    context_str = ""
    if text_contexts:
        citation_rule = " ทุกข้อเท็จจริงที่อ้างจากหลักฐานต้องใส่ citation เช่น [S1] ซึ่งต้องตรงกับ citation_id ที่มีอยู่จริง ห้ามสร้าง citation ใหม่หรืออ้างแหล่งข้อมูลที่ไม่มีในรายการ" if req.use_citations else " ไม่ต้องใส่ citation เพราะการทดลองนี้ปิด citation แต่ห้ามสร้างข้อเท็จจริงนอกหลักฐาน"
        context_str += "\n\n=== ข้อมูลจากคลังความรู้ (RAG Context) ===\n" + "\n---\n".join(text_contexts) + "\n========================================\n*ข้อบังคับสำคัญ*: ประวัติการสนทนาใช้เพื่อทำความเข้าใจคำถามเท่านั้น ไม่ใช่หลักฐานข้อเท็จจริง ตอบโดยอิงจากข้อมูลหลักฐานด้านบนเท่านั้น" + citation_rule + " หากหลักฐานไม่เพียงพอ ให้ตอบว่า 'ไม่พบข้อมูลที่เพียงพอจากแหล่งข้อมูล'"
    else:
        context_str = "\n\n=== ข้อมูลจากคลังความรู้ (RAG Context) ===\nไม่พบข้อมูลที่เพียงพอจากแหล่งข้อมูลสำหรับตอบคำถามนี้\n========================================\n*ข้อบังคับสำคัญ*: ห้ามใช้ความรู้ทั่วไป ห้ามเดา และให้ตอบเพียงว่า 'ไม่พบข้อมูลที่เพียงพอจากแหล่งข้อมูล'"
        
    if images_found:
        image_contexts = [
            ("[{citation_id}] source_name={source_name} document_id={document_id} "
             "chunk_id={chunk_id} image={name}").format(
                citation_id=image["citation_id"],
                source_name=image.get("source_name") or image.get("source"),
                document_id=image.get("document_id"),
                chunk_id=image.get("chunk_id"),
                name=image.get("name"),
            ) if req.use_citations else str(image.get("name"))
            for image in images_found if image.get("name")
        ]
        if image_contexts:
            context_str += "\n\n=== รูปภาพที่ค้นพบ ===\n" + "\n".join(image_contexts)

    return {
        "query": req.prompt,
        "original_query": req.prompt,
        "rewritten_query": rewritten_query,
        "query_rewrite_used_context": rewrite_used_context,
        "query_rewrite_latency_ms": rewrite_latency_ms,
        "query_rewrite_method": rewrite_method,
        "retrieval_latency_ms": retrieval_latency_ms,
        "reranking_latency_ms": reranking_latency_ms,
        "total_rag_latency_ms": round((time.perf_counter() - rewrite_started) * 1000, 3),
        "retrieval_score_semantics": "chroma_cosine_distance_lower_is_better",
        "evidence_max_distance": evidence_max_distance,
        "evidence_sufficient": evidence_sufficient,
        "evidence_reason": evidence_reason,
        "evidence_count": len(evidence_results),
        "rerank_score_semantics": "lexical_overlap_higher_is_better",
        "reranking_enabled": req.use_reranker,
        "candidate_count": candidate_k,
        "query_rewriting_enabled": req.use_query_rewriting,
        "citations_enabled": req.use_citations,
        "pipeline": [
            "conversation_memory_for_rewrite" if req.conversation_context else "no_conversation_context",
            "query_rewriting" if req.use_query_rewriting else "original_query",
            "chromadb_retrieval",
            "relevance_score",
            "reranking" if req.use_reranker else "retrieval_order",
            "citation_context" if req.use_citations else "plain_context",
            "grounded_generation",
        ],
        "grounding_rule": "Use retrieved evidence as the factual basis; if evidence is insufficient, say so; do not invent unsupported facts.",
        "context_str": context_str,
        "texts": results.get("texts", []),
        "images": images_found,
        "sources": retrieved_sources if req.use_citations else [],
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
