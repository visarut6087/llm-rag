#!/usr/bin/env python3
"""Run the Phase 9 A-G ablation matrix against the RAG API."""

import argparse
import csv
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


CONFIGURATIONS = [
    {"id": "A", "name": "baseline_rag", "use_citations": False, "use_reranker": False, "use_query_rewriting": False, "use_memory": False},
    {"id": "B", "name": "relevance_score", "use_citations": False, "use_reranker": False, "use_query_rewriting": False, "use_memory": False},
    {"id": "C", "name": "citations", "use_citations": True, "use_reranker": False, "use_query_rewriting": False, "use_memory": False},
    {"id": "D", "name": "reranking", "use_citations": True, "use_reranker": True, "use_query_rewriting": False, "use_memory": False},
    {"id": "E", "name": "query_rewriting", "use_citations": True, "use_reranker": True, "use_query_rewriting": True, "use_memory": False},
    {"id": "F", "name": "conversation_memory", "use_citations": True, "use_reranker": True, "use_query_rewriting": True, "use_memory": True},
    {"id": "G", "name": "full_rag", "use_citations": True, "use_reranker": True, "use_query_rewriting": True, "use_memory": True},
]


def post_json(url, payload):
    request = Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--dataset", default="results/evaluation/rag_evaluation_dataset.json")
    parser.add_argument("--output-dir", default="results/ablation")
    args = parser.parse_args()
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, errors = [], []

    for item in dataset["items"]:
        for config in CONFIGURATIONS:
            payload = {
                "prompt": item["question"], "n_results": 3,
                "use_citations": config["use_citations"],
                "use_reranker": config["use_reranker"],
                "use_query_rewriting": config["use_query_rewriting"],
                "conversation_context": item.get("conversation_context") if config["use_memory"] else [],
            }
            started = time.perf_counter()
            try:
                response = post_json(f"{args.base_url.rstrip('/')}/api/rag/query", payload)
                relevant = item.get("relevant_chunk")
                retrieved = [result.get("chunk_id") for result in response.get("texts", [])]
                rows.append({
                    "item_id": item["id"], "ablation_id": config["id"], "configuration": config["name"],
                    "answerable": item["answerable"], "relevant_chunk": relevant,
                    "hit_at_3": None if not item["answerable"] else float(relevant in retrieved),
                    "retrieved_chunk_ids": json.dumps(retrieved, ensure_ascii=False),
                    "pipeline": json.dumps(response.get("pipeline", []), ensure_ascii=False),
                    "retrieval_latency_ms": response.get("retrieval_latency_ms"),
                    "reranking_latency_ms": response.get("reranking_latency_ms"),
                    "query_rewrite_latency_ms": response.get("query_rewrite_latency_ms"),
                    "total_rag_latency_ms": response.get("total_rag_latency_ms"),
                    "client_latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "answer_relevance": None, "faithfulness": None,
                    "hallucination_rate": None, "citation_accuracy": None,
                })
            except (URLError, HTTPError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                errors.append({"item_id": item["id"], "ablation_id": config["id"], "configuration": config["name"], "error": str(exc)})

    status = "completed" if rows and not errors else ("partial" if rows else "not_run")
    result = {"status": status, "dataset": dataset["dataset_name"], "configurations": CONFIGURATIONS, "rows": rows, "errors": errors, "note": "Answer-quality fields remain null until the Phase 7 LLM evaluator is run; no unsupported improvements are claimed."}
    (output_dir / "ablation_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_rows = rows + errors
    fields = sorted({key for row in csv_rows for key in row}) if csv_rows else ["item_id", "ablation_id", "configuration", "error"]
    with (output_dir / "ablation_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(csv_rows)
    print(json.dumps({"status": status, "rows": len(rows), "errors": len(errors)}, ensure_ascii=False))
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
