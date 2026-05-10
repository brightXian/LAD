"""Minimal shuffle evaluation (delegates to methods)."""

from __future__ import annotations

import argparse

from tqdm import tqdm
from transformers import set_seed

from src.agd import analyze_candidate_tokens
from src.methods import build_method, run_agd_calibration
from src.models import load_pretrained
from src.utils import (
    build_shuffles,
    check_answer,
    cleanup_memory,
    load_eval_json,
    process_images,
    save_results,
    shuffle_images,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="multi-image eval")
    p.add_argument("--model-path", default="models/LLaVA-OneVision-1.5-8B-Instruct")
    p.add_argument(
        "--method",
        default="base",
        help="base|instruction|cot|agd|pride|sofa|fn|self_consistency|sc",
    )
    p.add_argument("--mode", choices=["random", "adversarial"], default="random")
    p.add_argument("--num-views", type=int, default=4)
    p.add_argument("--num-shuffles", type=int, default=3)
    p.add_argument("--max-samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--result-path", default="results")
    p.add_argument("--result-file", default=None)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--layer-range", type=int, nargs=2, default=[21, 35])
    p.add_argument("--calibration-samples", type=int, default=5)
    p.add_argument("--sharpen-power", type=float, default=5.0)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--sc-samples", type=int, default=10)
    p.add_argument("--sofa-sigma", type=float, default=0.5)
    p.add_argument("--sofa-every-n-layers", type=int, default=2)
    p.add_argument("--perm-avg-k", type=int, default=16)
    p.add_argument("--max-patches", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    meta = load_eval_json(args)
    model, processor = load_pretrained(args.model_path)
    cand = analyze_candidate_tokens(args.num_views, processor)

    bias_matrix = bias_scores = layers = None
    if args.method.lower() in ("agd", "lad", "ours", "pride"):
        bias_matrix, bias_scores, layers = run_agd_calibration(model, processor, meta, args, cand)

    method = build_method(
        args.method,
        model,
        processor,
        args,
        cand,
        bias_matrix=bias_matrix,
        bias_scores=bias_scores,
        target_layers=layers,
    )
    method.setup()
    shuffles = build_shuffles(args.num_shuffles, args.num_views)
    all_results: list[dict] = []
    try:
        for shuffle in tqdm(shuffles, desc="shuffles"):
            omap = {orig: pos for pos, orig in enumerate(shuffle)}
            rows: list[dict] = []
            ok = hit = 0
            n = min(args.max_samples, len(meta))
            sample_bar = tqdm(range(n), desc=f"eval {[i + 1 for i in shuffle]}", unit="sample")
            for sid in sample_bar:
                sample = meta[sid]
                paths = sample["image_ids"]
                caption = sample["caption"]
                pils = process_images(paths)
                shuffled_pils = shuffle_images(pils, shuffle)
                shuffled_paths = [paths[i] for i in shuffle]
                reply = method.predict(caption, shuffled_pils, shuffled_paths)
                gt_a, gen_a, is_ok, succ = check_answer(args, reply, sample, omap)
                if succ:
                    hit += 1
                if is_ok:
                    ok += 1
                done = sid + 1
                sample_bar.set_postfix(
                    acc=f"{ok / done:.4f}",
                    success=f"{hit / done:.4f}",
                    ok=f"{ok}/{done}",
                    hit=f"{hit}/{done}",
                )
                reply_preview = " ".join(reply.strip().split())
                if len(reply_preview) > 120:
                    reply_preview = f"{reply_preview[:117]}..."
                tqdm.write(
                    f"shuffle={[i + 1 for i in shuffle]} "
                    f"sample={sid + 1}/{n} "
                    f"gt={gt_a} pred={gen_a} "
                    f"correct={is_ok} success={succ} "
                    f"reply={reply_preview!r}"
                )
                rows.append(
                    {
                        "sample_id": sid,
                        "raw_response": reply.strip(),
                        "gt_answer": gt_a,
                        "generated_answer": gen_a,
                        "is_correct": is_ok,
                        "is_success": succ,
                    }
                )
                if sid % 5 == 0:
                    cleanup_memory()
            all_results.append(
                {
                    "shuffle_indices": [i + 1 for i in shuffle],
                    "accuracy": ok / max(1, n),
                    "success_rate": hit / max(1, n),
                    "predictions": rows,
                }
            )
    finally:
        method.cleanup()
    save_results(args, all_results)


if __name__ == "__main__":
    main()
