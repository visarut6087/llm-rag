# #!/usr/bin/env python3
# """
# Generate RAG answers only.

# This script does NOT judge/evaluate the answers.
# It records:
# - model parameters
# - generated answer
# - prompt/output/total token counts
# - latency per evaluation
# - total latency
# - RAG retrieval/reranking/query-rewrite latency
# - retrieved evidence and citations for later independent hallucination evaluation
# """

# import argparse
# import csv
# import json
# import re
# import time
# from pathlib import Path
# from urllib.error import HTTPError, URLError
# from urllib.request import Request, urlopen


# def load_json(path):
#     return json.loads(Path(path).read_text(encoding="utf-8"))


# def post_json(url, payload, timeout=120):
#     request = Request(
#         url,
#         data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
#         headers={"Content-Type": "application/json"},
#         method="POST",
#     )
#     with urlopen(request, timeout=timeout) as response:
#         return json.loads(response.read().decode("utf-8"))


# def ollama_chat(base_url, model, messages, temperature=0):
#     started = time.perf_counter()

#     response = post_json(
#         f"{base_url.rstrip('/')}/api/chat",
#         {
#             "model": model,
#             "messages": messages,
#             "stream": False,
#             "options": {"temperature": temperature},
#         },
#     )

#     client_latency_ms = round((time.perf_counter() - started) * 1000, 3)

#     message = response.get("message", {})
#     return {
#         "content": message.get("content", ""),
#         "prompt_eval_count": response.get("prompt_eval_count"),
#         "eval_count": response.get("eval_count"),
#         "total_tokens": (
#             (response.get("prompt_eval_count") or 0)
#             + (response.get("eval_count") or 0)
#             if response.get("prompt_eval_count") is not None
#             or response.get("eval_count") is not None
#             else None
#         ),
#         "ollama_total_duration_ns": response.get("total_duration"),
#         "ollama_load_duration_ns": response.get("load_duration"),
#         "ollama_prompt_eval_duration_ns": response.get("prompt_eval_duration"),
#         "ollama_eval_duration_ns": response.get("eval_duration"),
#         "client_latency_ms": client_latency_ms,
#     }


# def citation_info(answer, sources):
#     citations = sorted(set(re.findall(r"\[(S\d+)\]", answer or "")))
#     known = {source.get("citation_id"): source for source in sources}
#     return {
#         "citations_found": citations,
#         "known_citations": sorted(citations & known.keys()),
#     }


# def main():
#     parser = argparse.ArgumentParser(
#         description="Generate RAG answers only; no LLM judging."
#     )
#     parser.add_argument(
#         "--rag-url",
#         default="http://localhost:8000",
#         help="RAG API URL",
#     )
#     parser.add_argument(
#         "--ollama-url",
#         default="http://localhost:11434",
#         help="Ollama API URL",
#     )
#     parser.add_argument(
#         "--model",
#         required=True,
#         help="Ollama model name, e.g. qwen2.5:7b",
#     )
#     parser.add_argument(
#         "--dataset",
#         required=True,
#         help="Path to the evaluation dataset JSON",
#     )
#     parser.add_argument(
#         "--output-dir",
#         default=None,
#         help="Output directory. If omitted, uses results/evaluation/<model_name>",
#     )
#     parser.add_argument(
#         "--expected-items",
#         type=int,
#         default=None,
#         help="Optional expected number of dataset items",
#     )
#     parser.add_argument(
#         "--n-results",
#         type=int,
#         default=3,
#         help="Number of RAG results to retrieve",
#     )
#     parser.add_argument(
#         "--temperature",
#         type=float,
#         default=0,
#         help="Ollama temperature",
#     )
#     args = parser.parse_args()

#     dataset = load_json(args.dataset)

#     # Validate dataset before starting the expensive generation run.
#     items = dataset.get("items", [])
#     if not isinstance(items, list) or not items:
#         raise ValueError("Dataset must contain a non-empty 'items' list.")

#     ids = [item.get("id") for item in items]
#     if any(not item_id for item_id in ids):
#         raise ValueError("Every evaluation item must have an 'id'.")

#     duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
#     if duplicates:
#         raise ValueError(f"Duplicate evaluation item IDs: {duplicates}")

#     required_fields = {
#     "id",
#     "question",
#     "answer",
#     }

#     missing = []
#     for item in items:
#         absent = sorted(required_fields - set(item))
#         if absent:
#             missing.append(
#                 f"{item.get('id', '<unknown>')}: {', '.join(absent)}"
#             )

#     if missing:
#         raise ValueError("Missing dataset fields: " + " | ".join(missing))

#     if args.expected_items is not None and len(items) != args.expected_items:
#         raise ValueError(
#             f"Expected {args.expected_items} dataset items, "
#             f"but found {len(items)}."
#         )

#     model_dir_name = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model)
#     output_dir = Path(args.output_dir or f"results/evaluation/{model_dir_name}")
#     output_dir.mkdir(parents=True, exist_ok=True)

#     configurations = [
#         ("baseline_rag", False),
#     ]

#     total_evaluations = len(items) * len(configurations)

#     print(
#         f"Loaded evaluation dataset: {len(items)} questions "
#         f"({total_evaluations} generations)",
#         flush=True,
#     )
#     print(f"Model: {args.model}", flush=True)
#     print(f"Dataset: {args.dataset}", flush=True)
#     print(f"Output: {output_dir}", flush=True)
#     print(
#         f"Parameters: n_results={args.n_results}, "
#         f"temperature={args.temperature}, stream=False",
#         flush=True,
#     )

#     rows = []
#     errors = []
#     overall_started = time.perf_counter()

#     generation_index = 0

#     for index, item in enumerate(items, 1):
#         print(
#             f"\n[{index}/{len(items)}] "
#             f"({index / len(items) * 100:.1f}%) {item['id']} ...",
#             flush=True,
#         )

#         for configuration, use_reranker in configurations:
#             generation_index += 1
#             generation_started = time.perf_counter()

#             rag_payload = {
#                 "prompt": item["question"],
#                 "n_results": args.n_results,
#                 "use_reranker": use_reranker,
#             }

#             try:
#                 rag = post_json(
#                     f"{args.rag_url.rstrip('/')}/api/rag/query",
#                     rag_payload,
#                 )

#                 evidence = rag.get("context_str", "")
#                 sources = rag.get("sources", [])

#                 # Zero-shot grounding instruction.
#                 # No examples are supplied, so this is NOT few-shot.
#                 answer_prompt = (
#                     "ตอบคำถามโดยใช้หลักฐาน RAG ด้านล่างเท่านั้น "
#                     "ทุกข้อเท็จจริงต้องอ้าง [S#] ที่มีอยู่จริง "
#                     "หากหลักฐานไม่พอ ให้ตอบว่า "
#                     "'ไม่พบข้อมูลที่เพียงพอจากแหล่งข้อมูล'\n\n"
#                     f"คำถาม: {item['question']}\n"
#                     f"หลักฐาน:\n{evidence}"
#                 )

#                 llm = ollama_chat(
#                     args.ollama_url,
#                     args.model,
#                     [{"role": "user", "content": answer_prompt}],
#                     temperature=args.temperature,
#                 )

#                 generation_latency_ms = round(
#                     (time.perf_counter() - generation_started) * 1000, 3
#                 )

#                 citation_data = citation_info(llm["content"], sources)

#                 rows.append(
#                     {
#                         "item_id": item["id"],
#                         "configuration": configuration,
#                         "question": item["question"],
#                         "expected_answer": item["answer"],
#                         "generated_answer": llm["content"],
#                         "citations_found": citation_data["citations_found"],
#                         "known_citations": citation_data["known_citations"],
#                         "context_str": evidence,
#                         "sources": json.dumps(
#                             sources, ensure_ascii=False
#                         ),
#                         "model": args.model,
#                         "temperature": args.temperature,
#                         "n_results": args.n_results,
#                         "use_reranker": use_reranker,
#                         "prompt_tokens": llm["prompt_eval_count"],
#                         "output_tokens": llm["eval_count"],
#                         "total_tokens": llm["total_tokens"],
#                         "generation_latency_ms": generation_latency_ms,
#                         "ollama_client_latency_ms": llm["client_latency_ms"],
#                         "ollama_total_duration_ns": llm[
#                             "ollama_total_duration_ns"
#                         ],
#                         "ollama_load_duration_ns": llm[
#                             "ollama_load_duration_ns"
#                         ],
#                         "ollama_prompt_eval_duration_ns": llm[
#                             "ollama_prompt_eval_duration_ns"
#                         ],
#                         "ollama_eval_duration_ns": llm[
#                             "ollama_eval_duration_ns"
#                         ],
#                         "retrieval_latency_ms": rag.get(
#                             "retrieval_latency_ms"
#                         ),
#                         "reranking_latency_ms": rag.get(
#                             "reranking_latency_ms"
#                         ),
#                         "query_rewrite_latency_ms": rag.get(
#                             "query_rewrite_latency_ms"
#                         ),
#                         "total_rag_latency_ms": rag.get(
#                             "total_rag_latency_ms"
#                         ),
#                         # Hallucination is intentionally NOT scored here.
#                         # It requires an independent semantic judge.
#                         "hallucination": None,
#                     }
#                 )

#                 print(
#                     f"  {configuration}: "
#                     f"tokens={llm['total_tokens']} "
#                     f"(in={llm['prompt_eval_count']}, "
#                     f"out={llm['eval_count']}), "
#                     f"latency={generation_latency_ms} ms",
#                     flush=True,
#                 )

#             except (
#                 URLError,
#                 HTTPError,
#                 TimeoutError,
#                 OSError,
#                 ValueError,
#                 json.JSONDecodeError,
#             ) as exc:
#                 errors.append(
#                     {
#                         "item_id": item["id"],
#                         "configuration": configuration,
#                         "error": str(exc),
#                     }
#                 )
#                 print(
#                     f"  {configuration}: ERROR - {exc}",
#                     flush=True,
#                 )

#     total_latency_ms = round(
#         (time.perf_counter() - overall_started) * 1000, 3
#     )

#     def sum_numeric(field):
#         values = [
#             row[field]
#             for row in rows
#             if isinstance(row.get(field), (int, float))
#         ]
#         return sum(values) if values else None

#     summary = {
#         "model": args.model,
#         "dataset": str(args.dataset),
#         "output_dir": str(output_dir),
#         "dataset_name": dataset.get("dataset_name"),
#         "annotation_status": dataset.get("annotation_status"),
#         "total_questions": len(items),
#         "configurations": [name for name, _ in configurations],
#         "expected_generations": total_evaluations,
#         "successful_generations": len(rows),
#         "errors": len(errors),
#         "parameters": {
#             "model": args.model,
#             "temperature": args.temperature,
#             "stream": False,
#             "n_results": args.n_results,
#             "rag_url": args.rag_url,
#             "ollama_url": args.ollama_url,
#             "configurations": {
#                 "baseline_rag": {
#                     "use_reranker": False,
#                 },
#             },
#         },
#         "tokens_total": {
#             "prompt_tokens": sum_numeric("prompt_tokens"),
#             "output_tokens": sum_numeric("output_tokens"),
#             "total_tokens": sum_numeric("total_tokens"),
#         },
#         "latency_total_ms": total_latency_ms,
#         "latency_average_per_generation_ms": (
#             round(total_latency_ms / len(rows), 3) if rows else None
#         ),
#         "hallucination": (
#             "not_scored"
#         ),
#         "hallucination_note": (
#             "This generation-only script does not determine hallucination. "
#             "The generated answer, question, retrieved evidence, and sources "
#             "are saved so hallucination can be evaluated independently later."
#         ),
#         "rows": rows,
#         "errors_detail": errors,
#     }

#     (output_dir / "rag_generation.json").write_text(
#         json.dumps(summary, ensure_ascii=False, indent=2),
#         encoding="utf-8",
#     )

#     csv_rows = rows + errors
#     fieldnames = (
#         sorted({key for row in csv_rows for key in row})
#         if csv_rows
#         else ["item_id", "configuration", "error"]
#     )

#     with (output_dir / "rag_generation.csv").open(
#         "w", encoding="utf-8", newline=""
#     ) as handle:
#         writer = csv.DictWriter(handle, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(csv_rows)

#     print("\n===== SUMMARY =====", flush=True)
#     print(f"Model: {args.model}", flush=True)
#     print(
#         f"Generations: {len(rows)}/{total_evaluations}",
#         flush=True,
#     )
#     print(
#         f"Prompt tokens: {sum_numeric('prompt_tokens')}",
#         flush=True,
#     )
#     print(
#         f"Output tokens: {sum_numeric('output_tokens')}",
#         flush=True,
#     )
#     print(
#         f"Total tokens: {sum_numeric('total_tokens')}",
#         flush=True,
#     )
#     print(f"Total latency: {total_latency_ms} ms", flush=True)
#     print(
#         f"Average latency/generation: "
#         f"{summary['latency_average_per_generation_ms']} ms",
#         flush=True,
#     )
#     print("Hallucination: not scored", flush=True)
#     print(f"Saved: {output_dir}", flush=True)

#     return 0 if rows and not errors else (2 if errors else 1)


# if __name__ == "__main__":
#     raise SystemExit(main())



# # !python "/Users/foxixe/Downloads/Test llm /ragtest colab/gemini-ollama-gui/scripts/evaluate_llm_nodb.py" \
# #   --model qwen2.5:7b \
# #   --dataset "/Users/foxixe/Downloads/Test llm /ragtest colab/gemini-ollama-gui/scripts/dataset_for_train/setai20.json" \
# #   --output-dir "/Users/foxixe/Downloads/Test llm /ragtest colab/gemini-ollama-gui/results/evaluation/qwen2.5_7b"
