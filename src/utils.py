"""Shared utilities for evaluation and data prep."""

from __future__ import annotations

import gc
import json
import os
import random
import re
from typing import Any

import torch
from PIL import Image


def process_images(image_paths: list[str]) -> list[Image.Image]:
    return [Image.open(p).convert("RGB").resize((336, 336), Image.BICUBIC) for p in image_paths]


def shuffle_images(images: list[Image.Image], shuffle: list[int]) -> list[Image.Image]:
    return [images[i].resize((336, 336), Image.BICUBIC) for i in shuffle]


def encode_hf_multimodal(
    processor: Any,
    messages: list[dict[str, Any]],
    pil_images: list[Image.Image],
    device: torch.device,
    dtype: torch.dtype | None = None,
    max_patches: int | None = None,
) -> dict[str, torch.Tensor]:
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    num_images = max(len(pil_images), 1)
    max_num_tiles = max_patches or max(1, min(12, 28 // num_images))
    batch = processor(
        text=[prompt],
        images=pil_images,
        padding=True,
        return_tensors="pt",
        max_patches=max_num_tiles,
    )
    return {
        k: v.to(device=device, dtype=dtype if dtype is not None and v.is_floating_point() else None)
        for k, v in batch.items()
    }


def find_image_token_ranges(input_ids: torch.Tensor, tokenizer_like: Any) -> list[tuple[int, int]]:
    """Return one (start, end) slice per image; end is exclusive (Python slice)."""
    tok = getattr(tokenizer_like, "tokenizer", tokenizer_like)
    row = input_ids[0].tolist()
    vs = tok.convert_tokens_to_ids("<|vision_start|>")
    ve = tok.convert_tokens_to_ids("<|vision_end|>")
    unk = tok.unk_token_id
    ranges: list[tuple[int, int]] = []
    if vs != unk and ve != unk:
        i = 0
        while i < len(row):
            if row[i] == vs:
                start = i
                j = i + 1
                while j < len(row) and row[j] != ve:
                    j += 1
                if j < len(row):
                    ranges.append((start, j + 1))
                    i = j + 1
                    continue
            i += 1
        if ranges:
            return ranges

    img_s = tok.convert_tokens_to_ids("<img>")
    img_e = tok.convert_tokens_to_ids("</img>")
    if img_s == unk or img_e == unk:
        return ranges
    i = 0
    while i < len(row):
        if row[i] == img_s:
            start = i
            j = i + 1
            while j < len(row) and row[j] != img_e:
                j += 1
            if j < len(row):
                ranges.append((start, j + 1))
                i = j + 1
                continue
        i += 1
    return ranges


def parse_digit_answer(response: str, num_views: int, method: str = "") -> int | None:
    if method.lower() in ("sc", "self_consistency"):
        matches = re.findall(r"[Tt]he answer is\s+(\d+)", response)
        valid_matches = [int(n) for n in matches if 1 <= int(n) <= num_views]
        if valid_matches:
            return valid_matches[-1]
    nums = [int(x) for x in re.findall(r"\d+", response.strip())]
    valid = [n for n in nums if 1 <= n <= num_views]
    return valid[-1] if valid else None


def check_answer(
    args: Any,
    response: str,
    sample: dict[str, Any],
    orig_to_shuffled: dict[int, int],
) -> tuple[int, int, bool, bool]:
    """Return (gt_answer_1based, generated_1based, is_correct, is_success)."""
    num_views = args.num_views
    gt = int(sample["index"])
    gt_shuffled = orig_to_shuffled[gt] + 1
    gen = parse_digit_answer(response, num_views, getattr(args, "method", ""))
    if gen is None:
        return gt_shuffled, -1, False, False
    return gt_shuffled, gen, gen == gt_shuffled, True


def save_results(args: Any, shuffle_results: list[dict[str, Any]]) -> None:
    os.makedirs(args.result_path, exist_ok=True)
    model_name = os.path.basename(os.path.normpath(args.model_path))
    name = args.result_file or (
        f"eval_{model_name}_{args.method}_{args.mode}_"
        f"views{args.num_views}_shuffles{args.num_shuffles}_samples{args.max_samples}.json"
    )
    path = os.path.join(args.result_path, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(shuffle_results, f, indent=2)
    print(f"Saved results -> {path}")


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_shuffles(num_shuffles: int, num_candidates: int) -> list[list[int]]:
    base = list(range(num_candidates))
    out = [base]
    seen = {tuple(base)}
    while len(out) < num_shuffles:
        perm = tuple(random.sample(base, len(base)))
        if perm in seen:
            continue
        seen.add(perm)
        out.append(list(perm))
    return out


def load_eval_json(args: Any) -> list[dict[str, Any]]:
    path = f"data/eval/annotations_{args.num_views}_1_{args.mode}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)
