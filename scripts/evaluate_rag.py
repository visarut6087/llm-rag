#!/usr/bin/env python3
"""Evaluate retrieval independently from answer generation.

The script calls the running RAG API and writes JSON/CSV results. It never
creates relevance labels: those must already exist in the input dataset.
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def query_api(base_url, payload):
    request = Request(
        f"{base_url.rstrip('/')}/api/rag/query",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    body["client_total_latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return body


def metric_values(response, item, k):
    if not item["answerable"] or not item.get("relevant_source") or not item.get("relevant_chunk"):
        return {"recall_at_k": None, "precision_at_k": None, "hit_rate_at_k": None, "mrr": None, "ndcg_at_k": None}

    hits = [
        result for result in response.get("texts", [])[:k]
        if result.get("source") == item["relevant_source"]
        and result.get("chunk_id") == item["relevant_chunk"]
    ]
    rank = hits[0].get("rank") if hits else None
    return {
        "recall_at_k": 1.0 if hits else 0.0,
        "precision_at_k": len(hits) / k,
        "hit_rate_at_k": 1.0 if hits else 0.0,
        "mrr": 1.0 / rank if rank else 0.0,
        "ndcg_at_k": 1.0 / math.log2(rank + 1) if rank else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--dataset", default="results/evaluation/rag_evaluation_dataset.json")
    parser.add_argument("--output-dir", default="results/evaluation")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    dataset = load_json(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configurations = [
        ("baseline_rag", False, False),
        ("relevance_score", False, False),
        ("reranking", True, False),
        ("query_rewriting", False, True),
        ("conversation_memory", True, True),
    ]
    rows = []
    run_errors = []

    for item in dataset["items"]:
        for name, use_reranker, use_context in configurations:
            payload = {
                "prompt": item["question"],
                "n_results": args.k,
                "use_reranker": use_reranker,
                "conversation_context": item.get("conversation_context") if use_context else [],
            }
            try:
                response = query_api(args.base_url, payload)
                metrics = metric_values(response, item, args.k)
                rows.append({
                    "item_id": item["id"],
                    "configuration": name,
                    "answerable": item["answerable"],
                    "original_query": response.get("original_query"),
                    "rewritten_query": response.get("rewritten_query"),
                    "retrieved_chunk_ids": json.dumps([r.get("chunk_id") for r in response.get("texts", [])], ensure_ascii=False),
                    **metrics,
                    "retrieval_latency_ms": response.get("retrieval_latency_ms"),
                    "reranking_latency_ms": response.get("reranking_latency_ms"),
                    "query_rewrite_latency_ms": response.get("query_rewrite_latency_ms"),
                    "client_total_latency_ms": response.get("client_total_latency_ms"),
                })
            except (URLError, HTTPError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                run_errors.append({"item_id": item["id"], "configuration": name, "error": str(exc)})

    status = "completed" if rows and not run_errors else ("partial" if rows else "not_run")
    result = {
        "status": status,
        "dataset": dataset["dataset_name"],
        "annotation_status": dataset["annotation_status"],
        "k": args.k,
        "configurations": [name for name, _, _ in configurations],
        "rows": rows,
        "run_errors": run_errors,
        "note": "Metrics are null for manually annotated unanswerable items because no relevant source/chunk exists.",
    }
    (output_dir / "rag_evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_rows = rows + run_errors
    fieldnames = sorted({key for row in csv_rows for key in row}) if csv_rows else ["item_id", "configuration", "error"]
    with (output_dir / "rag_evaluation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(json.dumps({"status": status, "rows": len(rows), "errors": len(run_errors)}, ensure_ascii=False))
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
