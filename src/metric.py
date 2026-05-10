"""Compact evaluation metrics for saved result JSON files."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="summarize evaluation result metrics")
    parser.add_argument("--result", required=True, help="result JSON file or directory")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--output-file", default=None)
    return parser.parse_args()


def parse_answer(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def load_result(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("shuffle_evaluations", [])


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def summarize(path: Path) -> str:
    shuffles = load_result(path)
    accuracies = [float(item.get("accuracy", 0.0)) for item in shuffles]
    success_rates = [
        float(
            item.get("success_rate", item.get("success", 0) / max(1, item.get("total_sample", 1)))
        )
        for item in shuffles
    ]

    position_correct: dict[int, int] = defaultdict(int)
    position_total: dict[int, int] = defaultdict(int)
    prediction_total: dict[int, int] = defaultdict(int)
    total = 0
    correct = 0

    for shuffle in shuffles:
        for pred in shuffle.get("predictions", []):
            gt = parse_answer(pred.get("gt_answer"))
            generated = parse_answer(pred.get("generated_answer"))
            if gt is not None:
                position_total[gt] += 1
                total += 1
            if generated is not None:
                prediction_total[generated] += 1
            if pred.get("is_correct", False):
                correct += 1
                if gt is not None:
                    position_correct[gt] += 1

    positions = sorted(position_total)
    position_acc = [
        position_correct[position] / position_total[position]
        for position in positions
        if position_total[position] > 0
    ]
    rstd = std(position_acc)

    lines = [
        f"file {path.name}",
        f"shuffles {len(shuffles)}",
        f"samples {total}",
        f"accuracy {correct / max(1, total):.4f}",
        f"accuracy_mean {sum(accuracies) / max(1, len(accuracies)):.4f}",
        f"accuracy_std {std(accuracies):.4f}",
        f"success_mean {sum(success_rates) / max(1, len(success_rates)):.4f}",
        f"rstd {rstd:.4f}",
        "positions",
    ]

    for position in positions:
        count = position_total[position]
        lines.append(
            f"{position} gt {count} pred {prediction_total[position]} "
            f"acc {position_correct[position] / max(1, count):.4f}"
        )
    return "\n".join(lines)


def result_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.json"))


def main() -> None:
    args = parse_args()
    result_path = Path(args.result)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = [summarize(path) for path in result_files(result_path)]
    text = "\n\n".join(reports)
    output_name = args.output_file or f"{result_path.stem}_metrics.txt"
    output_path = output_dir / output_name
    output_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
