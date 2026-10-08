#!/usr/bin/env python3
"""Generate direct Ollama answers without retrieval, using the shared RAG result schema."""

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "scripts" / "dataset_for_train"
RESULTS_DIR = ROOT / "results"
DEFAULT_DATASET = DATASET_DIR / "sarabun-questions50.json"


def resolve_dataset(dataset_arg):
    supplied = Path(dataset_arg).expanduser()
    candidates = [supplied] if supplied.is_absolute() else [Path.cwd() / supplied, DATASET_DIR / supplied]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Dataset not found: {dataset_arg}")


def load_dataset(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else data.get("items", []) if isinstance(data, dict) else []
    if not isinstance(items, list) or not items:
        raise ValueError("Dataset must be a non-empty JSON list or an object containing 'items'.")

    ids = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict) or not item.get("question"):
            raise ValueError(f"Dataset item {index} must contain a question.")
        if "id" not in item:
            raise ValueError(f"Dataset item {index} must contain an id.")
        ids.append(item["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Dataset contains duplicate question IDs.")
    return items


def ollama_chat(base_url, model, question, temperature=0):
    prompt = (
        "Answer the following question in Thai using the model's own knowledge only. "
        "No documents or external context are provided. If you do not know, say so.\n\n"
        f"Question: {question}"
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature},
    }
    request = Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urlopen(request, timeout=600) as response:
        result = json.loads(response.read().decode("utf-8"))
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    message = result.get("message", {})
    prompt_tokens = result.get("prompt_eval_count")
    output_tokens = result.get("eval_count")
    total_tokens = (
        (prompt_tokens or 0) + (output_tokens or 0)
        if prompt_tokens is not None or output_tokens is not None
        else None
    )
    return {
        "answer": message.get("content", ""),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "latency_ms": latency_ms,
        "ollama_total_duration_ns": result.get("total_duration"),
        "ollama_load_duration_ns": result.get("load_duration"),
        "ollama_prompt_eval_duration_ns": result.get("prompt_eval_duration"),
        "ollama_eval_duration_ns": result.get("eval_duration"),
    }


def main():
    # Windows terminals may default to a legacy code page that cannot print Thai.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Generate direct Ollama answers without using RAG, in the shared evaluation format."
    )
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Dataset filename or JSON path")
    parser.add_argument("--model", default="qwen2.5:7b", help="Ollama model name")
    parser.add_argument("--agent", default="default", help="Run label, saved in the result metadata")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N questions")
    parser.add_argument("--output-dir", default=None, help="Output directory; a timestamped run folder is used by default")
    args = parser.parse_args()

    dataset_path = resolve_dataset(args.dataset)
    items = load_dataset(dataset_path)
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be greater than zero")
        items = items[:args.limit]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model).strip("_") or "model"
    safe_agent = re.sub(r"[^A-Za-z0-9._-]+", "_", args.agent).strip("_") or "default"
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else RESULTS_DIR / "baseline_no_rag" / safe_model / f"{safe_agent}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    errors = []
    run_started = time.perf_counter()
    print(f"Dataset: {dataset_path} ({len(items)} questions)", flush=True)
    print(f"Model: {args.model} | RAG: disabled | Output: {output_dir}", flush=True)

    for index, item in enumerate(items, start=1):
        question = item["question"]
        item_started = time.perf_counter()
        print(f"\n[{index}/{len(items)}] id={item['id']}: {question[:100]}", flush=True)
        try:
            generated = ollama_chat(args.ollama_url, args.model, question, args.temperature)
            latency_ms = round((time.perf_counter() - item_started) * 1000, 3)
            rows.append({
                "id": item["id"],
                "answer": generated["answer"],
                "citations": "[]",
                "citation_count": 0,
                "hallucination": "",
                "prompt_tokens": generated["prompt_tokens"],
                "output_tokens": generated["output_tokens"],
                "total_tokens": generated["total_tokens"],
                "latency_ms": latency_ms,
                "ollama_latency_ms": generated["latency_ms"],
                "rag_latency_ms": None,
                "model": args.model,
                "agent": args.agent,
                "temperature": args.temperature,
                "n_results": None,
                "question": question,
                "context": "",
                "sources": "[]",
            })
            print(
                f"  OK tokens={generated['total_tokens']} latency={latency_ms} ms",
                flush=True,
            )
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"id": item["id"], "question": question, "error": str(exc)})
            print(f"  ERROR: {exc}", flush=True)

    total_run_latency_ms = round((time.perf_counter() - run_started) * 1000, 3)

    def numeric_sum(field):
        values = [row[field] for row in rows if isinstance(row.get(field), (int, float))]
        return round(sum(values), 3) if values else None

    summary = {
        "model": args.model,
        "agent": args.agent,
        "parameters": {
            "model": args.model,
            "agent": args.agent,
            "temperature": args.temperature,
            "n_results": None,
            "use_reranker": False,
            "use_rag": False,
            "stream": False,
            "ollama_url": args.ollama_url,
        },
        "dataset": {
            "path": str(dataset_path),
            "name": dataset_path.name,
            "questions_requested": len(items),
            "questions_completed": len(rows),
            "errors": len(errors),
        },
        "hallucination": "",
        "token": {
            "prompt_tokens_total": numeric_sum("prompt_tokens"),
            "output_tokens_total": numeric_sum("output_tokens"),
            "total_tokens": numeric_sum("total_tokens"),
        },
        "latency": {
            "total_item_latency_ms": numeric_sum("latency_ms"),
            "total_run_latency_ms": total_run_latency_ms,
        },
        "rows": rows,
        "errors_detail": errors,
    }

    json_path = output_dir / "answers.json"
    csv_path = output_dir / "answers.csv"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = list(rows[0].keys()) if rows else [
        "id", "answer", "citations", "citation_count", "hallucination",
        "prompt_tokens", "output_tokens", "total_tokens", "latency_ms",
        "ollama_latency_ms", "rag_latency_ms", "model", "agent", "temperature",
        "n_results", "question", "context", "sources",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 60)
    print(f"Model: {args.model} | Agent: {args.agent} | RAG: disabled")
    print(f"Questions: {len(items)} | Completed: {len(rows)} | Errors: {len(errors)}")
    print(f"Total tokens: {numeric_sum('total_tokens')}")
    print(f"Total item latency: {numeric_sum('latency_ms')} ms")
    print(f"Total run latency: {total_run_latency_ms} ms")
    print(f"JSON: {json_path}\nCSV: {csv_path}")
    return 0 if rows and not errors else (2 if errors else 1)


if __name__ == "__main__":
    raise SystemExit(main())
