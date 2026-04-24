#!/usr/bin/env python3
"""
Run conditional probability query tests from a JSON input file.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from conditional_cooccurrence import CooccurrenceStore

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def surprisal_bits(probability: float) -> float:
    if probability <= 0.0:
        return float("inf")
    return -math.log2(probability)


def load_input(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "queries" not in data or not isinstance(data["queries"], list):
        raise ValueError("Input JSON must contain a list field 'queries'.")
    return data


def parse_sentence_words(sentence: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(sentence)]


def get_sentence_count(store: CooccurrenceStore) -> int:
    raw = store.get_meta("sentence_count")
    if raw is None:
        return 0
    try:
        return int(json.loads(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0


def sentence_chain_probabilities(store: CooccurrenceStore, sentence: str) -> Dict[str, Any]:
    words = parse_sentence_words(sentence)
    if not words:
        return {
            "probabilities": [],
            "surprisals": [],
            "surprisal_sum": 0.0,
            "surprisal_variance": 0.0,
            "status": "error",
        }

    probabilities: List[float] = []
    surprisals: List[float] = []
    sentence_count = get_sentence_count(store)

    for i, target in enumerate(words):
        if i == 0:
            denom = sentence_count
            numer = store.get_unigram_count(target)
        elif i == 1:
            prev = words[i - 1]
            denom = store.get_unigram_count(prev)
            numer = len(store.get_pair_intersection(target, prev))
        else:
            prev1 = words[i - 1]
            prev2 = words[i - 2]
            context = store.get_pair_intersection(prev2, prev1)
            denom = len(context)
            numer = len(context & store.get_posting(target))

        prob = (numer / denom) if denom else 0.0
        probabilities.append(prob)
        surprisals.append(surprisal_bits(prob))

    surprisal_sum = sum(surprisals)
    mean_surprisal = surprisal_sum / len(surprisals)
    surprisal_variance = sum((s - mean_surprisal) ** 2 for s in surprisals) / len(
        surprisals
    )
    return {
        "probabilities": probabilities,
        "surprisals": surprisals,
        "surprisal_sum": surprisal_sum,
        "surprisal_variance": surprisal_variance,
        "status": "ok",
    }


def run_queries(db_path: Path, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    store = CooccurrenceStore(db_path)
    results: List[Dict[str, Any]] = []
    try:
        for q in queries:
            qid = q.get("id", "unknown")
            qtype = q.get("type")
            a = str(q.get("a", "")).lower()
            b = str(q.get("b", "")).lower()
            c = q.get("c")

            if not a or not b:
                results.append(
                    {
                        "id": qid,
                        "status": "error",
                        "message": "Query must include non-empty 'a' and 'b'.",
                    }
                )
                continue

            if qtype == "pair":
                numer = len(store.get_pair_intersection(a, b))
                denom = store.get_unigram_count(b)
                prob = (numer / denom) if denom else 0.0
                results.append(
                    {
                        "id": qid,
                        "type": "pair",
                        "query": f"P({a}|{b})",
                        "a": a,
                        "b": b,
                        "numerator_count": numer,
                        "denominator_count": denom,
                        "probability": prob,
                        "surprisal_bits": surprisal_bits(prob),
                        "category": q.get("category", []),
                        "status": "ok",
                    }
                )
            elif qtype == "triple":
                if not c:
                    results.append(
                        {
                            "id": qid,
                            "status": "error",
                            "message": "Triple query must include field 'c'.",
                        }
                    )
                    continue
                c = str(c).lower()
                bc = store.get_pair_intersection(b, c)
                denom = len(bc)
                numer = len(bc & store.get_posting(a))
                prob = (numer / denom) if denom else 0.0
                results.append(
                    {
                        "id": qid,
                        "type": "triple",
                        "query": f"P({a}|{b},{c})",
                        "a": a,
                        "b": b,
                        "c": c,
                        "numerator_count": numer,
                        "denominator_count": denom,
                        "probability": prob,
                        "surprisal_bits": surprisal_bits(prob),
                        "category": q.get("category", []),
                        "status": "ok",
                    }
                )
            else:
                results.append(
                    {
                        "id": qid,
                        "status": "error",
                        "message": "Query type must be either 'pair' or 'triple'.",
                    }
                )
    finally:
        store.close()
    return results


def run_sentence_mode(db_path: Path, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    store = CooccurrenceStore(db_path)
    results: List[Dict[str, Any]] = []
    try:
        for q in queries:
            qid = q.get("id", "unknown")
            qtype = q.get("type")
            sentence = str(q.get("sentence", ""))

            if qtype != "sentence":
                results.append(
                    {
                        "id": qid,
                        "type": str(qtype),
                        "sentence": sentence,
                        "probabilities": [],
                        "surprisals": [],
                        "status": "error",
                    }
                )
                continue

            chain = sentence_chain_probabilities(store, sentence)
            results.append(
                {
                    "id": qid,
                    "type": "sentence",
                    "sentence": sentence,
                    "probabilities": chain["probabilities"],
                    "surprisals": chain["surprisals"],
                    "surprisal_sum": chain["surprisal_sum"],
                    "surprisal_variance": chain["surprisal_variance"],
                    "status": chain["status"],
                }
            )
    finally:
        store.close()
    return results


def save_sentence_surprisal_plots(
    results: List[Dict[str, Any]], output_dir: Path
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]
        import seaborn as sns  # type: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires seaborn and matplotlib. Install with: pip install seaborn matplotlib"
        ) from exc

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in results:
        if item.get("type") != "sentence" or item.get("status") != "ok":
            continue
        group_id = str(item.get("id", "unknown"))
        grouped.setdefault(group_id, []).append(item)

    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: List[Path] = []

    sns.set_theme(style="whitegrid")
    for sentence_id, items in grouped.items():
        if not items:
            continue

        fig, ax = plt.subplots(figsize=(12, 7))
        palette = sns.color_palette("tab10", n_colors=max(1, len(items)))
        variance_annotation_lines: List[tuple[str, Any]] = []

        for idx, item in enumerate(items):
            surprisals = item.get("surprisals", [])
            if not isinstance(surprisals, list) or not surprisals:
                continue

            words = parse_sentence_words(str(item.get("sentence", "")))
            if len(words) != len(surprisals):
                # Defensive fallback: annotate only aligned prefix if lengths diverge.
                words = words[: len(surprisals)]

            x_positions = list(range(1, len(surprisals) + 1))
            label = str(item.get("sentence", ""))
            color = palette[idx % len(palette)]

            ax.plot(x_positions, surprisals, marker="o", linewidth=2, color=color, label=label)
            variance = item.get("surprisal_variance")
            if isinstance(variance, (int, float)):
                variance_annotation_lines.append(
                    (
                        f"var({label}) = {float(variance):.4f}",
                        color,
                    )
                )

            for word_idx, word in enumerate(words):
                y_value = surprisals[word_idx]
                ax.annotate(
                    word,
                    (x_positions[word_idx], y_value),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=9,
                    color=color,
                )

        ax.set_title(f"Sentence ID {sentence_id}: surprisal by word position")
        ax.set_xlabel("Word position")
        ax.set_ylabel("Surprisal (bits)")
        ax.set_xticks(
            list(range(1, max(len(item.get("surprisals", [])) for item in items) + 1))
        )
        ax.legend(title="Sentence variant", loc="best", fontsize=9)
        if variance_annotation_lines:
            for line_idx, (line_text, line_color) in enumerate(variance_annotation_lines):
                ax.text(
                    0.01,
                    0.02 + (line_idx * 0.05),
                    line_text,
                    transform=ax.transAxes,
                    va="bottom",
                    ha="left",
                    color=line_color,
                    fontsize=9,
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "white",
                        "edgecolor": line_color,
                        "alpha": 0.75,
                    },
                )
        fig.tight_layout()

        safe_id = re.sub(r"[^a-zA-Z0-9._-]+", "_", sentence_id).strip("_") or "unknown"
        out_path = output_dir / f"sentence_surprisal_{safe_id}.pdf"
        fig.savefig(out_path)
        plt.close(fig)
        written_paths.append(out_path)

    return written_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read JSON query fixture and print conditional probability results."
    )
    parser.add_argument(
        "--input",
        default="data/query_test_input.json",
        help="Path to JSON file containing test queries",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Optional override path to SQLite artifact (otherwise uses input JSON db_path)",
    )
    parser.add_argument(
        "--mode",
        choices=["queries", "sentence-chain"],
        default="queries",
        help="queries: existing pair/triple behavior; sentence-chain: only sentence chain outputs",
    )
    parser.add_argument(
        "--plot-sentence-surprisal",
        action="store_true",
        help="When in sentence-chain mode, save surprisal-vs-position plots grouped by sentence id.",
    )
    parser.add_argument(
        "--plot-output-dir",
        default="artifacts/plots",
        help="Directory for sentence surprisal plots (used with --plot-sentence-surprisal).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    data = load_input(input_path)
    db_path = Path(args.db) if args.db else Path(data.get("db_path", "artifacts/cooccurrence.sqlite"))
    queries = data["queries"]

    if args.mode == "sentence-chain":
        results = run_sentence_mode(db_path, queries)
        if args.plot_sentence_surprisal:
            output_paths = save_sentence_surprisal_plots(results, Path(args.plot_output_dir))
            print(
                f"Wrote {len(output_paths)} plot(s) to {Path(args.plot_output_dir)}",
                file=sys.stderr,
            )
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    results = run_queries(db_path, queries)
    payload = {
        "input_file": str(input_path),
        "db_path": str(db_path),
        "query_count": len(queries),
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
