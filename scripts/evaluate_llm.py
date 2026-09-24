#!/usr/bin/env python3
"""Evaluate generated RAG answers separately from retrieval metrics.

Generation and judging are delegated to Ollama. If either service is
unavailable, the output keeps null metric values rather than inferring scores.
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


METRICS = (
    "faithfulness",
    "answer_relevance",
    "context_relevance",
    "citation_correctness",
    "hallucination_rate",
    "abstention_correctness",
)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def post_json(url, payload, timeout=120):
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ollama_chat(base_url, model, messages):
    response = post_json(
        f"{base_url.rstrip('/')}/api/chat",
        {"model": model, "messages": messages, "stream": False, "options": {"temperature": 0}},
    )
    return response.get("message", {}).get("content", "")


def parse_judgement(text):
    match = re.search(r"\{[\s\S]*\}", text or "")
    if not match:
        raise ValueError("judge response did not contain JSON")
    data = json.loads(match.group(0))
    result = {metric: data.get(metric) for metric in METRICS}
    result["rationale"] = data.get("rationale")
    return result


def citation_check(answer, sources, item):
    citations = set(re.findall(r"\[(S\d+)\]", answer or ""))
    known = {source.get("citation_id"): source for source in sources}
    valid = [known[citation] for citation in citations if citation in known]
    expected = [
        source for source in valid
        if source.get("source_name") == item.get("relevant_source")
        and source.get("chunk_id") == item.get("relevant_chunk")
    ]
    return {
        "citations_found": sorted(citations),
        "known_citations": sorted(citations & known.keys()),
        "expected_source_cited": bool(expected),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-url", default="http://localhost:8000")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--dataset", default="results/evaluation/llm_evaluation_dataset.json")
    parser.add_argument("--output-dir", default="results/evaluation")
    args = parser.parse_args()

    dataset = load_json(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configurations = [
        ("baseline_rag", False, False),
        ("reranking", True, False),
        ("conversation_memory", True, True),
    ]
    rows = []
    errors = []

    # for item in dataset["items"]:
    #     current = len(rows) + len(errors) + 1
    total = len(dataset["items"])

    for index, item in enumerate(dataset["items"], 1):
        print(f"\n[{index}/{total}] ({index / total * 100:.1f}%) กำลังประเมิน {item['id']} ...", flush=True)

        # print(f"[{current}/{len(dataset['items'])}] กำลังประเมิน {item.get('id', '')} ...", flush=True)

        for configuration, use_reranker, use_context in configurations:
            rag_payload = {
                "prompt": item["question"],
                "n_results": 3,
                "use_reranker": use_reranker,
                "conversation_context": item.get("conversation_context") if use_context else [],
            }
            try:
                rag = post_json(f"{args.rag_url.rstrip('/')}/api/rag/query", rag_payload)
                evidence = rag.get("context_str", "")
                answer_prompt = (
                    "ตอบคำถามโดยใช้หลักฐาน RAG ด้านล่างเท่านั้น ทุกข้อเท็จจริงต้องอ้าง [S#] "
                    "ที่มีอยู่จริง หากหลักฐานไม่พอ ให้ตอบว่า 'ไม่พบข้อมูลที่เพียงพอจากแหล่งข้อมูล'\n\n"
                    f"คำถาม: {item['question']}\nหลักฐาน:\n{evidence}"
                )
                answer = ollama_chat(args.ollama_url, args.model, [{"role": "user", "content": answer_prompt}])
                citation_info = citation_check(answer, rag.get("sources", []), item)

                judge_prompt = (
                    "คุณเป็นผู้ประเมินคำตอบ RAG ตอบเป็น JSON เท่านั้น โดยให้ค่า 0 หรือ 1 "
                    "สำหรับแต่ละ metric: faithfulness, answer_relevance, context_relevance, "
                    "citation_correctness, hallucination_rate, abstention_correctness "
                    "(hallucination_rate=1 หมายถึงมี hallucination; metric อื่น 1 หมายถึงผ่าน) "
                    "และฟิลด์ rationale สั้น ๆ\n\n"
                    f"คำถาม: {item['question']}\nคำตอบคาดหมาย: {item['expected_answer']}\n"
                    f"ต้อง abstain: {item['abstention_required']}\nหลักฐาน: {evidence}\n"
                    f"คำตอบที่สร้าง: {answer}\nข้อมูล citation ที่ตรวจได้: {json.dumps(citation_info, ensure_ascii=False)}"
                )
                judgement = parse_judgement(ollama_chat(args.ollama_url, args.model, [{"role": "user", "content": judge_prompt}]))
                rows.append({
                    "item_id": item["id"],
                    "configuration": configuration,
                    "answerable": item["answerable"],
                    "question_type": item["question_type"],
                    "generated_answer": answer,
                    **citation_info,
                    **judgement,
                    "retrieval_latency_ms": rag.get("retrieval_latency_ms"),
                    "reranking_latency_ms": rag.get("reranking_latency_ms"),
                    "query_rewrite_latency_ms": rag.get("query_rewrite_latency_ms"),
                })
            except (URLError, HTTPError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append({"item_id": item["id"], "configuration": configuration, "error": str(exc)})

    status = "completed" if rows and not errors else ("partial" if rows else "not_run")
    result = {
        "status": status,
        "dataset": dataset["dataset_name"],
        "annotation_status": dataset["annotation_status"],
        "judge_model": args.model,
        "metrics": list(METRICS),
        "rows": rows,
        "errors": errors,
        "note": "Scores are null when generation or judging is unavailable; retrieval quality is reported separately by rag_evaluation.json.",
    }
    (output_dir / "llm_evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_rows = rows + errors
    fieldnames = sorted({key for row in csv_rows for key in row}) if csv_rows else ["item_id", "configuration", "error"]
    with (output_dir / "llm_evaluation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({"status": status, "rows": len(rows), "errors": len(errors)}, ensure_ascii=False))
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
