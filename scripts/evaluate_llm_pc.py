#!/usr/bin/env python3
"""
Generate RAG answers.

Workflow:
Dataset question
    -> RAG retrieves context/evidence
    -> Question + RAG context prompt
    -> Ollama/AI agent
    -> new answer
    -> JSON/CSV output

Requirements:
- Dataset can be selected at runtime with --dataset.
- Dataset files are resolved from the fixed dataset_for_train directory when
  only a filename is supplied.
- Output is created automatically under the fixed results directory.
- Every successful item contains: id, answer, hallucination(blank),
  token counts and latency.
- Summary contains total tokens and total latency across all successful items.
- --agent can be specified at runtime and is sent to the RAG API when present.
"""

import argparse
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "scripts" / "dataset_for_train"
RESULTS_DIR = BASE_DIR / "results"


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


def ollama_chat(base_url, model, messages, temperature=0):
    started = time.perf_counter()

    response = post_json(
        f"{base_url.rstrip('/')}/api/chat",
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
    )

    client_latency_ms = round((time.perf_counter() - started) * 1000, 3)
    message = response.get("message", {})

    prompt_tokens = response.get("prompt_eval_count")
    output_tokens = response.get("eval_count")

    return {
        "content": message.get("content", ""),
        "prompt_eval_count": prompt_tokens,
        "eval_count": output_tokens,
        "total_tokens": (
            (prompt_tokens or 0) + (output_tokens or 0)
            if prompt_tokens is not None or output_tokens is not None
            else None
        ),
        "client_latency_ms": client_latency_ms,
        "ollama_total_duration_ns": response.get("total_duration"),
        "ollama_load_duration_ns": response.get("load_duration"),
        "ollama_prompt_eval_duration_ns": response.get("prompt_eval_duration"),
        "ollama_eval_duration_ns": response.get("eval_duration"),
    }


def resolve_dataset(dataset_arg):
    """
    Accept either:
      --dataset sarabun-questions-50.json
    or:
      --dataset /full/path/to/file.json

    A filename is automatically resolved inside DATASET_DIR.
    """
    supplied = Path(dataset_arg).expanduser()

    candidates = []
    if supplied.is_absolute():
        candidates.append(supplied)
    else:
        candidates.extend([
            Path.cwd() / supplied,
            DATASET_DIR / supplied,
        ])

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "ไม่พบ Dataset:\n"
        f"  ที่ระบุ: {dataset_arg}\n"
        f"  ตรวจใน: {DATASET_DIR}\n"
        "ตัวอย่าง: --dataset sarabun-questions-50.json"
    )


def normalize_items(dataset):
    if isinstance(dataset, list):
        items = dataset
    elif isinstance(dataset, dict):
        items = dataset.get("items", [])
    else:
        raise ValueError(
            "Dataset JSON ต้องเป็น list หรือ object ที่มี 'items'"
        )

    if not isinstance(items, list) or not items:
        raise ValueError("Dataset ไม่มีข้อมูล")

    return items


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "output"

def extract_citations(answer):
    """
    Extract citation IDs such as [S1], [S2], [S10]
    from the LLM-generated answer.
    """
    if not answer:
        return []

    citations = re.findall(r"\[S\d+\]", answer, flags=re.IGNORECASE)

    # Normalize to uppercase and remove duplicates while preserving order.
    seen = set()
    result = []

    for citation in citations:
        citation = citation.upper()
        if citation not in seen:
            seen.add(citation)
            result.append(citation)

    return result

def main():
    parser = argparse.ArgumentParser(
        description="Question + RAG context -> Ollama answer generator (Windows/RTX 3070 Ti)"
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Dataset filename in scripts/dataset_for_train, "
            "or full path to JSON"
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Ollama model name, e.g. qwen2.5:7b",
    )
    parser.add_argument(
        "--agent",
        default="default",
        help="Agent name to use for this run; sent to RAG API and recorded in output",
    )
    parser.add_argument(
        "--rag-url",
        default="http://localhost:8000",
        help="RAG API URL",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama API URL",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=3,
        help="Number of RAG results",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Ollama temperature",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N questions",
    )
    # parser.add_argument(
    #     "--use-reranker",
    #     action="store_true",
    #     help="Ask RAG API to use reranker",
    # )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional custom output directory. Default: project/results",
    )

    args = parser.parse_args()

    dataset_path = resolve_dataset(args.dataset)
    dataset = load_json(dataset_path)
    items = normalize_items(dataset)

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit ต้องมากกว่า 0")
        items = items[: args.limit]

    if not items:
        raise ValueError("ไม่มีคำถามสำหรับรัน")

    # Keep the input IDs exactly as supplied by the dataset.
    ids = [item.get("id") for item in items]
    if any(item_id is None for item_id in ids):
        raise ValueError("ทุกข้อใน Dataset ต้องมี id")

    duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
    if duplicates:
        raise ValueError(f"พบ id ซ้ำใน Dataset: {duplicates}")

    for item in items:
        if "question" not in item:
            raise ValueError(f"ข้อ {item.get('id')} ไม่มี field 'question'")

    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else RESULTS_DIR
    )
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = safe_name(dataset_path.stem)
    model_name = safe_name(args.model)
    agent_name = safe_name(args.agent)

    # Each run gets its own directory automatically.
    output_dir = output_root / f"{dataset_name}__{model_name}__{agent_name}__{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RAG → Question + Context → AI Answer")
    print("=" * 70)
    print(f"Dataset : {dataset_path}")
    print(f"Questions: {len(items)}")
    print(f"Model   : {args.model}")
    print(f"Agent   : {args.agent}")
    print(f"RAG     : {args.rag_url}")
    print(f"Ollama  : {args.ollama_url}")
    print(
        f"Params  : n_results={args.n_results}, "
        f"temperature={args.temperature}, "
        # f"use_reranker={args.use_reranker}"
    )
    print(f"Output  : {output_dir}")
    print("=" * 70)

    rows = []
    errors = []
    overall_started = time.perf_counter()

    for index, item in enumerate(items, 1):
        item_id = item["id"]
        question = item["question"]
        item_started = time.perf_counter()

        print(
            f"\n[{index}/{len(items)}] id={item_id}: {question[:100]}",
            flush=True,
        )

        try:
            # 1) Send ONLY the question to RAG.
            rag_payload = {
                "prompt": question,
                "n_results": args.n_results,
                # "use_reranker": args.use_reranker,
            }

            # The RAG server can use the selected agent if its API supports it.
            if args.agent:
                rag_payload["agent"] = args.agent

            rag_started = time.perf_counter()

            rag = post_json(
                f"{args.rag_url.rstrip('/')}/api/rag/query",
                rag_payload,
            )

            rag_client_latency_ms = round(
                (time.perf_counter() - rag_started) * 1000, 3
            )

            context = rag.get("context_str", "")
            sources = rag.get("sources", [])

            # 2) Build the final AI prompt from Question + RAG Context.
            answer_prompt = (
                "คุณเป็น AI สำหรับตอบคำถามโดยอ้างอิงข้อมูลจาก RAG\n"
                "ให้ตอบคำถามโดยใช้ข้อมูลจาก Context ด้านล่างเป็นหลัก\n"
                "ห้ามแต่งข้อมูลที่ไม่มีอยู่ใน Context\n"
                "ทุกข้อเท็จจริงที่ตอบต้องอ้างอิงแหล่งข้อมูลด้วยรูปแบบ [S#]\n"
                "โดยใช้เฉพาะ [S#] ที่มีอยู่ใน Context เท่านั้น\n"
                "ห้ามสร้างหมายเลข [S#] ที่ไม่มีอยู่ใน Context\n"
                "หาก Context ไม่มีข้อมูลเพียงพอ ให้ตอบว่า "
                "'ไม่พบข้อมูลที่เพียงพอจากแหล่งข้อมูล'\n\n"
                f"Question:\n{question}\n\n"
                f"Context:\n{context}\n\n"
                "Answer:"
            )

            # 3) Send Question + Context prompt to AI.
            llm = ollama_chat(
                args.ollama_url,
                args.model,
                [{"role": "user", "content": answer_prompt}],
                temperature=args.temperature,
            )
            answer = llm["content"]
            citations = extract_citations(answer)

            total_item_latency_ms = round(
                (time.perf_counter() - item_started) * 1000, 3
            )

            row = {
                # Exactly the same ID as the input.
                "id": item_id,

                # New answer generated after Question + RAG Context -> AI.
                "answer": answer,

                # Citation IDs generated by the LLM, e.g. ["S1", "S3"].
                "citations": json.dumps(citations, ensure_ascii=False),

                # Number of unique citations generated by the LLM.
                "citation_count": len(citations),

                "hallucination": "",

                # Per-item token information.
                "prompt_tokens": llm["prompt_eval_count"],
                "output_tokens": llm["eval_count"],
                "total_tokens": llm["total_tokens"],

                # Per-item latency.
                "latency_ms": total_item_latency_ms,
                "ollama_latency_ms": llm["client_latency_ms"],
                "rag_latency_ms": rag_client_latency_ms,

                # Run configuration / parameters.
                "model": args.model,
                "agent": args.agent,
                "temperature": args.temperature,
                "n_results": args.n_results,
                # "use_reranker": args.use_reranker,

                # Keep these for traceability.
                "question": question,
                "context": context,
                "sources": json.dumps(sources, ensure_ascii=False),
            }

            rows.append(row)

            print(
                f"  ✓ tokens={llm['total_tokens']} "
                f"latency={total_item_latency_ms} ms",
                flush=True,
            )

        except (
            URLError,
            HTTPError,
            TimeoutError,
            OSError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            errors.append(
                {
                    "id": item_id,
                    "question": question,
                    "error": str(exc),
                }
            )
            print(f"  ✗ ERROR: {exc}", flush=True)

    total_latency_ms = round(
        (time.perf_counter() - overall_started) * 1000, 3
    )

    def numeric_sum(field):
        values = [
            row[field]
            for row in rows
            if isinstance(row.get(field), (int, float))
        ]
        return round(sum(values), 3) if values else None

    successful = len(rows)

    # Requested run-level metrics.
    total_prompt_tokens = numeric_sum("prompt_tokens")
    total_output_tokens = numeric_sum("output_tokens")
    total_tokens = numeric_sum("total_tokens")
    total_item_latency_ms = numeric_sum("latency_ms")

    summary = {
        "model": args.model,
        "agent": args.agent,
        "parameters": {
            "model": args.model,
            "agent": args.agent,
            "temperature": args.temperature,
            "n_results": args.n_results,
            # "use_reranker": args.use_reranker,
            "stream": False,
            "rag_url": args.rag_url,
            "ollama_url": args.ollama_url,
        },
        "dataset": {
            "path": str(dataset_path),
            "name": dataset_path.name,
            "questions_requested": len(items),
            "questions_completed": successful,
            "errors": len(errors),
        },
        "hallucination": "",
        "token": {
            "prompt_tokens_total": total_prompt_tokens,
            "output_tokens_total": total_output_tokens,
            "total_tokens": total_tokens,
        },
        "latency": {
            "total_item_latency_ms": total_item_latency_ms,
            "total_run_latency_ms": total_latency_ms,
        },
        "rows": rows,
        "errors_detail": errors,
    }

    json_path = output_dir / "answers.json"
    csv_path = output_dir / "answers.csv"

    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fieldnames = (
        list(rows[0].keys())
        if rows
        else ["id", "answer", "hallucination"]
    )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model                  : {args.model}")
    print(f"Agent                  : {args.agent}")
    print(f"Questions              : {len(items)}")
    print(f"Completed              : {successful}")
    print(f"Errors                 : {len(errors)}")
    print(f"Hallucination          : (blank)")
    print(f"Prompt tokens (total)  : {total_prompt_tokens}")
    print(f"Output tokens (total)  : {total_output_tokens}")
    print(f"Tokens (total)         : {total_tokens}")
    print(f"Latency items (total)  : {total_item_latency_ms} ms")
    print(f"Latency run (total)    : {total_latency_ms} ms")
    print(f"JSON                   : {json_path}")
    print(f"CSV                    : {csv_path}")
    print("=" * 70)

    # Return non-zero if at least one item failed.
    return 0 if rows and not errors else (2 if errors else 1)


if __name__ == "__main__":
    raise SystemExit(main())
