#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Condition Matrix - Pure Mathematical / Character Trigram Evaluation

1. เลือกโฟลเดอร์จาก results ด้วยหน้าต่างเลือกโฟลเดอร์
2. หา answers.csv และ JSON metadata ในโฟลเดอร์นั้นอัตโนมัติ
3. เลือก Gold Dataset (.json)
4. เปรียบเทียบ AI answer กับ Gold answer ด้วย Character N-gram
   ค่าเริ่มต้น = Trigram (3-gram)
5. คำนวณ Precision / Recall / F1 / Accuracy / Grounding เป็น %
6. ห้ามใช้ LLM / API / semantic judge
7. ดึง model / total_tokens / latency_ms จาก JSON metadata
8. สร้าง condition_matrix.csv, condition_matrix.json,
   condition_summary.json

นิยาม:
P = multiset ของ N-gram จาก AI answer
G = multiset ของ N-gram จาก Gold answer
matched = sum(min(count_P[g], count_G[g]))

Precision = matched / |P|
Recall    = matched / |G|
F1        = 2PR / (P+R)
Accuracy  = matched / max(|P|, |G|)

Grounding = overlap(AI answer N-grams, RAG context N-grams)
            / จำนวน N-grams ของ AI answer

หมายเหตุ:
ใช้ Character N-gram เพราะภาษาไทยไม่ได้เว้นวรรคระหว่างทุกคำ
และไม่ใช้ LLM เป็นผู้ตัดสิน
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"
DEFAULT_GOLD_DIR = SCRIPT_DIR / "dataset_for_train"

DEFAULT_N = 3
ANSWER_CSV_NAMES = ("answers.csv",)
GOLD_NAME_PATTERNS = ("sarabun-questions50.json", "sarabun-questions100.json")

CITATION_RE = re.compile(r"\[S\d+\]", re.IGNORECASE)


# ============================================================
# TEXT / N-GRAM
# ============================================================

def clean_text(text: Any) -> str:
    if text is None:
        return ""

    text = unicodedata.normalize("NFC", str(text))
    text = CITATION_RE.sub(" ", text)

    # ลบ whitespace เพื่อให้ Character N-gram ทำงานกับภาษาไทยได้ต่อเนื่อง
    text = re.sub(r"\s+", "", text)

    return text


def char_ngrams(text: Any, n: int = DEFAULT_N) -> Counter[str]:
    text = clean_text(text)

    if not text:
        return Counter()

    if len(text) <= n:
        return Counter([text])

    return Counter(text[i:i + n] for i in range(len(text) - n + 1))


def overlap_count(a: Counter[str], b: Counter[str]) -> int:
    return sum((a & b).values())


def prf_accuracy(pred: Counter[str], gold: Counter[str]) -> dict[str, float]:
    pred_total = sum(pred.values())
    gold_total = sum(gold.values())
    matched = overlap_count(pred, gold)

    precision = matched / pred_total if pred_total else 0.0
    recall = matched / gold_total if gold_total else 0.0

    if precision + recall:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 1.0 if not pred_total and not gold_total else 0.0

    # ใช้ symmetric N-gram overlap เป็น Accuracy
    denominator = max(pred_total, gold_total)
    accuracy = matched / denominator if denominator else 1.0

    return {
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "accuracy": accuracy * 100,
        "matched_ngrams": matched,
        "answer_ngrams": pred_total,
        "gold_ngrams": gold_total,
    }


def grounding_score(answer: Any, context: Any, n: int = DEFAULT_N) -> float:
    answer_ngrams = char_ngrams(answer, n)
    context_ngrams = char_ngrams(context, n)

    total = sum(answer_ngrams.values())

    if total == 0:
        return 0.0

    return overlap_count(answer_ngrams, context_ngrams) / total * 100


# ============================================================
# FILE
# ============================================================

def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_gold(data: Any) -> dict[str, dict[str, Any]]:
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        if isinstance(data.get("items"), list):
            items = data["items"]
        elif isinstance(data.get("data"), list):
            items = data["data"]
        elif isinstance(data.get("rows"), list):
            items = data["rows"]
        else:
            items = [data]
    else:
        raise ValueError("Gold JSON ต้องเป็น list หรือ object")

    result = {}

    for item in items:
        if isinstance(item, dict) and item.get("id") is not None:
            result[str(item["id"])] = item

    return result


def find_csv(folder: Path) -> Path:
    candidates = sorted(folder.glob("*.csv"))

    if not candidates:
        raise FileNotFoundError(
            f"ไม่พบไฟล์ CSV ในโฟลเดอร์:\n{folder}"
        )

    # ให้ answers.csv มี priority สูงสุด
    for name in ANSWER_CSV_NAMES:
        for path in candidates:
            if path.name.lower() == name.lower():
                return path

    # ถ้าไม่มี answers.csv ให้หา CSV ที่มี id + answer
    for path in candidates:
        try:
            rows = read_csv(path)
            if rows and "id" in rows[0] and "answer" in rows[0]:
                return path
        except Exception:
            pass

    return candidates[0]


def find_json_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.json"))


# ============================================================
# JSON METADATA
# ============================================================

def iter_dicts(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from iter_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_dicts(value)


def to_number(value: Any):
    if value is None or value == "":
        return None

    try:
        x = float(value)
        return int(x) if x.is_integer() else x
    except (TypeError, ValueError):
        return None


def load_metadata(json_files: list[Path], csv_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    รองรับ JSON ได้หลายรูปแบบ:
    - answers.json ที่มีข้อมูลรายข้อ
    - summary.json ที่มีข้อมูลระดับ run
    - JSON ที่มี summary / parameters
    """

    objects = []

    for path in json_files:
        try:
            data = read_json(path)
            objects.extend(iter_dicts(data))
        except Exception:
            continue

    per_id = {}

    for obj in objects:
        if obj.get("id") is not None:
            item_id = str(obj["id"])

            if any(
                key in obj
                for key in ("model", "total_tokens", "latency_ms")
            ):
                per_id[item_id] = obj

    run_meta = {}

    for obj in objects:
        if "model" in obj and any(
            key in obj
            for key in (
                "total_tokens",
                "total_latency_ms",
                "latency_ms",
                "total_item_latency_ms",
            )
        ):
            run_meta = obj
            break

    # รองรับกรณี summary อยู่ใน object
    if not run_meta:
        for obj in objects:
            if isinstance(obj.get("summary"), dict):
                summary = obj["summary"]

                if "model" in summary or "total_tokens" in summary:
                    run_meta = summary
                    break

    csv_model = next(
        (row.get("model") for row in csv_rows if row.get("model")),
        None,
    )

    csv_total_tokens = None
    csv_latency = None

    totals = [
        to_number(row.get("total_tokens"))
        for row in csv_rows
        if to_number(row.get("total_tokens")) is not None
    ]

    latencies = [
        to_number(row.get("latency_ms"))
        for row in csv_rows
        if to_number(row.get("latency_ms")) is not None
    ]

    if totals:
        csv_total_tokens = sum(totals)

    if latencies:
        csv_latency = sum(latencies)

    parameters = run_meta.get("parameters", {})
    if not isinstance(parameters, dict):
        parameters = {}

    model = (
        run_meta.get("model")
        or parameters.get("model")
        or csv_model
    )

    total_tokens = (
        to_number(run_meta.get("total_tokens"))
        or to_number(run_meta.get("total_item_tokens"))
        or csv_total_tokens
    )

    latency_ms = (
        to_number(run_meta.get("total_latency_ms"))
        or to_number(run_meta.get("total_item_latency_ms"))
        or to_number(run_meta.get("latency_ms"))
        or csv_latency
    )

    return {
        "model": model,
        "total_tokens": total_tokens,
        "latency_ms": latency_ms,
        "per_id": per_id,
        "run_meta": run_meta,
    }


# ============================================================
# GUI
# ============================================================

def choose_results_folder() -> Path:
    """
    พยายามใช้ GUI ก่อน
    ถ้า Tk/Tcl ใช้งานไม่ได้ ให้ fallback เป็นเมนูเลือกผ่าน PowerShell/console
    เพื่อให้โปรแกรมยังทำงานได้แม้ Python ติดตั้ง Tcl/Tk ไม่สมบูรณ์
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        initial_dir = (
            RESULTS_DIR
            if RESULTS_DIR.is_dir()
            else PROJECT_DIR
        )

        folder = filedialog.askdirectory(
            title="เลือกโฟลเดอร์ผลลัพธ์จาก results",
            initialdir=str(initial_dir),
            mustexist=True,
        )

        root.destroy()

        if not folder:
            raise SystemExit("ยกเลิกการเลือกโฟลเดอร์")

        return Path(folder).resolve()

    except Exception as exc:
        print("\nไม่สามารถเปิดหน้าต่างเลือกโฟลเดอร์ได้")
        print(f"สาเหตุ: {exc}")
        print("\nใช้โหมด Console แทน...")

        if not RESULTS_DIR.is_dir():
            # ถ้าไม่มี results ตามตำแหน่งมาตรฐาน ให้ผู้ใช้พิมพ์ path เอง
            raw = input("\nกรุณาใส่ path ของโฟลเดอร์ results: ").strip().strip('"')
            folder = Path(raw).expanduser().resolve()

            if not folder.is_dir():
                raise FileNotFoundError(
                    f"ไม่พบโฟลเดอร์: {folder}"
                )

            return folder

        # แสดงเฉพาะโฟลเดอร์ผลลัพธ์ระดับแรก
        folders = sorted(
            [
                p for p in RESULTS_DIR.iterdir()
                if p.is_dir()
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not folders:
            raw = input(
                "\nไม่พบ subfolder ใน results "
                "กรุณาใส่ path ของ result folder: "
            ).strip().strip('"')

            folder = Path(raw).expanduser().resolve()

            if not folder.is_dir():
                raise FileNotFoundError(
                    f"ไม่พบโฟลเดอร์: {folder}"
                )

            return folder

        print("\nโฟลเดอร์ใน results:")
        for i, folder in enumerate(folders, 1):
            print(f"  [{i}] {folder.name}")

        while True:
            choice = input(
                f"\nเลือกหมายเลขโฟลเดอร์ [1-{len(folders)}]: "
            ).strip()

            try:
                index = int(choice) - 1
                if 0 <= index < len(folders):
                    return folders[index].resolve()
            except ValueError:
                pass

            print("กรุณาเลือกหมายเลขให้ถูกต้อง")


def choose_gold(default_path: Path | None = None) -> Path:
    """
    พยายามใช้ GUI ก่อน
    ถ้า Tk/Tcl ใช้งานไม่ได้ ให้ fallback เป็น console
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        if default_path and default_path.is_file():
            initial_dir = default_path.parent
        elif DEFAULT_GOLD_DIR.is_dir():
            initial_dir = DEFAULT_GOLD_DIR
        else:
            initial_dir = PROJECT_DIR

        path = filedialog.askopenfilename(
            title="เลือก Gold Dataset (.json)",
            initialdir=str(initial_dir),
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
        )

        root.destroy()

        if not path:
            raise SystemExit("ยกเลิกการเลือก Gold Dataset")

        return Path(path).resolve()

    except Exception as exc:
        print("\nไม่สามารถเปิดหน้าต่างเลือก Gold Dataset ได้")
        print(f"สาเหตุ: {exc}")
        print("\nใช้โหมด Console แทน...")

        candidates = []

        if default_path and default_path.is_file():
            candidates.append(default_path.resolve())

        if DEFAULT_GOLD_DIR.is_dir():
            for p in sorted(DEFAULT_GOLD_DIR.glob("*.json")):
                if p.resolve() not in candidates:
                    candidates.append(p.resolve())

        if candidates:
            print("\nGold Dataset ที่พบ:")
            for i, p in enumerate(candidates, 1):
                print(f"  [{i}] {p}")

            while True:
                choice = input(
                    f"\nเลือกหมายเลข Gold [1-{len(candidates)}]: "
                ).strip()

                try:
                    index = int(choice) - 1
                    if 0 <= index < len(candidates):
                        return candidates[index]
                except ValueError:
                    pass

                print("กรุณาเลือกหมายเลขให้ถูกต้อง")

        raw = input(
            "\nไม่พบ Gold Dataset อัตโนมัติ "
            "กรุณาใส่ path ของไฟล์ .json: "
        ).strip().strip('"')

        path = Path(raw).expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"ไม่พบ Gold Dataset: {path}"
            )

        return path


# ============================================================
# EVALUATION
# ============================================================

def evaluate_rows(
    csv_rows: list[dict[str, Any]],
    gold: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
    n: int,
) -> list[dict[str, Any]]:

    results = []

    for row in csv_rows:
        item_id = str(row.get("id", "")).strip()
        gold_item = gold.get(item_id)

        answer = row.get("answer", "")
        context = row.get("context", "")

        per_id = metadata["per_id"].get(item_id, {})

        model = (
            per_id.get("model")
            or metadata.get("model")
            or row.get("model")
        )

        total_tokens = (
            to_number(per_id.get("total_tokens"))
            if per_id.get("total_tokens") is not None
            else None
        )

        latency_ms = (
            to_number(per_id.get("latency_ms"))
            if per_id.get("latency_ms") is not None
            else None
        )

        # ถ้า JSON เป็น run-level ให้ใช้ run metadata
        if total_tokens is None:
            total_tokens = metadata.get("total_tokens")

        if latency_ms is None:
            latency_ms = metadata.get("latency_ms")

        result = {
            "id": item_id,
            "question": row.get("question", ""),
            "model": model,
            "total_tokens": total_tokens,
            "latency_ms": latency_ms,
            "status": "SCORED" if gold_item else "MISSING_GOLD",
        }

        if not gold_item:
            result.update({
                "gold_answer": "",
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "accuracy": 0.0,
                "grounding": round(
                    grounding_score(answer, context, n),
                    4,
                ),
                "matched_ngrams": 0,
                "answer_ngrams": sum(
                    char_ngrams(answer, n).values()
                ),
                "gold_ngrams": 0,
            })

            results.append(result)
            continue

        gold_answer = gold_item.get("answer", "")

        pred_ngrams = char_ngrams(answer, n)
        gold_ngrams = char_ngrams(gold_answer, n)

        scores = prf_accuracy(
            pred_ngrams,
            gold_ngrams,
        )

        result.update({
            "gold_answer": gold_answer,
            "precision": round(scores["precision"], 4),
            "recall": round(scores["recall"], 4),
            "f1": round(scores["f1"], 4),
            "accuracy": round(scores["accuracy"], 4),
            "grounding": round(
                grounding_score(answer, context, n),
                4,
            ),
            "matched_ngrams": scores["matched_ngrams"],
            "answer_ngrams": scores["answer_ngrams"],
            "gold_ngrams": scores["gold_ngrams"],
        })

        results.append(result)

    return results


def average_metric(
    results: list[dict[str, Any]],
    field: str,
) -> float:

    values = [
        float(row[field])
        for row in results
        if row.get("status") == "SCORED"
        and row.get(field) is not None
    ]

    return sum(values) / len(values) if values else 0.0


def build_summary(
    results,
    selected_folder,
    csv_path,
    gold_path,
    json_files,
    n,
    metadata,
):

    scored = [
        row
        for row in results
        if row.get("status") == "SCORED"
    ]

    return {
        "method": "pure_mathematical_character_ngram",
        "no_llm_judge": True,

        "ngram": {
            "type": "character",
            "n": n,
            "name": f"character-{n}-gram",
            "normalization": (
                "Unicode NFC + remove whitespace + remove [S#]"
            ),
        },

        "files": {
            "results_folder": str(selected_folder),
            "answers_csv": str(csv_path),
            "gold_json": str(gold_path),
            "json_metadata_files": [
                str(path)
                for path in json_files
            ],
        },

        "dataset": {
            "rows_in_csv": len(results),
            "scored": len(scored),
            "missing_gold": len(results) - len(scored),
        },

        "metrics_percent": {
            "precision": round(
                average_metric(results, "precision"),
                4,
            ),
            "recall": round(
                average_metric(results, "recall"),
                4,
            ),
            "f1": round(
                average_metric(results, "f1"),
                4,
            ),
            "accuracy": round(
                average_metric(results, "accuracy"),
                4,
            ),
            "grounding": round(
                average_metric(results, "grounding"),
                4,
            ),
        },

        "run_metadata_from_json": {
            "model": metadata.get("model"),
            "total_tokens": metadata.get("total_tokens"),
            "latency_ms": metadata.get("latency_ms"),
        },
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Condition Matrix - "
            "Character N-gram / Trigram / No LLM Judge"
        )
    )

    parser.add_argument(
        "--results-folder",
        default=None,
        help="กำหนด result folder โดยตรง; ถ้าไม่ใส่จะเปิด GUI",
    )

    parser.add_argument(
        "--gold",
        default=None,
        help="Path ของ Gold JSON; ถ้าไม่ใส่จะเปิด GUI",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory",
    )

    parser.add_argument(
        "--ngram",
        type=int,
        default=DEFAULT_N,
        help="Character N-gram size; default = 3",
    )

    args = parser.parse_args()

    if args.ngram < 1:
        raise ValueError("--ngram ต้องมีค่า >= 1")

    # 1. เลือก result folder
    if args.results_folder:
        selected_folder = (
            Path(args.results_folder)
            .expanduser()
            .resolve()
        )
    else:
        selected_folder = choose_results_folder()

    if not selected_folder.is_dir():
        raise FileNotFoundError(
            f"ไม่พบโฟลเดอร์:\n{selected_folder}"
        )

    # 2. หา CSV / JSON อัตโนมัติ
    csv_path = find_csv(selected_folder)
    json_files = find_json_files(selected_folder)

    # 3. Gold
    if args.gold:
        gold_path = (
            Path(args.gold)
            .expanduser()
            .resolve()
        )
    else:
        conventional = []

        for name in GOLD_NAME_PATTERNS:
            path = DEFAULT_GOLD_DIR / name

            if path.is_file():
                conventional.append(path)

        gold_path = choose_gold(
            conventional[0]
            if conventional
            else None
        )

    if not gold_path.is_file():
        raise FileNotFoundError(
            f"ไม่พบ Gold Dataset:\n{gold_path}"
        )

    # 4. Load
    csv_rows = read_csv(csv_path)
    gold = normalize_gold(
        read_json(gold_path)
    )

    metadata = load_metadata(
        json_files,
        csv_rows,
    )

    # 5. Output
    if args.output_dir:
        output_dir = (
            Path(args.output_dir)
            .expanduser()
            .resolve()
        )
    else:
        output_dir = (
            selected_folder
            / "condition_matrix"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("CONDITION MATRIX - PURE MATHEMATICAL N-GRAM")
    print("=" * 72)
    print(f"Results folder : {selected_folder}")
    print(f"CSV            : {csv_path.name}")
    print(
        "JSON metadata  : "
        + (
            ", ".join(
                path.name
                for path in json_files
            )
            if json_files
            else "NONE"
        )
    )
    print(f"Gold           : {gold_path}")
    print(
        f"N-gram         : character {args.ngram}-gram"
    )
    print(f"Rows           : {len(csv_rows)}")
    print(f"Gold items     : {len(gold)}")
    print(
        f"Model          : {metadata.get('model')}"
    )
    print(
        f"Total tokens   : {metadata.get('total_tokens')}"
    )
    print(
        f"Latency (ms)   : {metadata.get('latency_ms')}"
    )
    print("=" * 72)

    results = evaluate_rows(
        csv_rows,
        gold,
        metadata,
        args.ngram,
    )

    summary = build_summary(
        results,
        selected_folder,
        csv_path,
        gold_path,
        json_files,
        args.ngram,
        metadata,
    )

    # ========================================================
    # CSV
    # ========================================================

    output_csv = (
        output_dir
        / "condition_matrix.csv"
    )

    fields = [
        "id",
        "question",
        "model",
        "total_tokens",
        "latency_ms",
        "status",
        "gold_answer",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "grounding",
        "matched_ngrams",
        "answer_ngrams",
        "gold_ngrams",
    ]

    with output_csv.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in results:

            output_row = {}

            for field in fields:

                value = row.get(
                    field,
                    "",
                )

                if field in {
                    "precision",
                    "recall",
                    "f1",
                    "accuracy",
                    "grounding",
                } and isinstance(
                    value,
                    (int, float),
                ):
                    value = f"{value:.4f}%"

                output_row[field] = value

            writer.writerow(output_row)

    # ========================================================
    # JSON
    # ========================================================

    output_json = (
        output_dir
        / "condition_matrix.json"
    )

    output_summary = (
        output_dir
        / "condition_summary.json"
    )

    payload = {
        "method": (
            "pure_mathematical_character_ngram"
        ),
        "no_llm_judge": True,
        "ngram": (
            f"character-{args.ngram}-gram"
        ),
        "summary": summary,
        "results": results,
    }

    output_json.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    output_summary.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    metrics = summary["metrics_percent"]

    print("\n" + "=" * 72)
    print("SUMMARY (%)")
    print("=" * 72)
    print(
        f"Precision : {metrics['precision']:.2f}%"
    )
    print(
        f"Recall    : {metrics['recall']:.2f}%"
    )
    print(
        f"F1        : {metrics['f1']:.2f}%"
    )
    print(
        f"Accuracy  : {metrics['accuracy']:.2f}%"
    )
    print(
        f"Grounding : {metrics['grounding']:.2f}%"
    )
    print("-" * 72)
    print(
        f"Model     : {metadata.get('model')}"
    )
    print(
        f"Tokens    : {metadata.get('total_tokens')}"
    )
    print(
        f"Latency   : {metadata.get('latency_ms')} ms"
    )
    print(
        f"\nOutput    : {output_dir}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
