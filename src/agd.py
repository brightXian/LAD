"""Attention-Guided Debiasing (AGD / former LAD): logits + attention helpers (simplified, same logic)."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from src.utils import (
    encode_hf_multimodal,
    find_image_token_ranges,
    process_images,
)


def _calibration_messages(caption: str, num_views: int) -> list[dict[str, Any]]:
    prompt = (
        f"Given {num_views} images indexed from 1 to {num_views}, "
        "identify the image that best matches the provided caption."
        f" Respond with the index number only and nothing else.\nCaption: {caption}\nAnswer: "
    )
    return [
        {
            "role": "user",
            "content": [
                *[{"type": "image"} for _ in range(num_views)],
                {"type": "text", "text": prompt},
            ],
        }
    ]


def analyze_candidate_tokens(num_candidates: int, processor: Any) -> dict[str, Any]:
    tok = getattr(processor, "tokenizer", processor)
    eos = tok.eos_token_id
    if eos is None:
        eos = tok.convert_tokens_to_ids("</s>")
        if eos is None or eos == tok.unk_token_id:
            eos = tok.convert_tokens_to_ids("<|endoftext|>")

    single_ids: list[int] = []
    single_idx: list[int] = []
    multi_groups: dict[int, tuple[str, list[tuple[int, int]]]] = {}
    suffix_ids: set[int] = set()

    for i in range(1, num_candidates + 1):
        s = str(i)
        if i <= 9:
            tid = tok.convert_tokens_to_ids(s)
            if tid is None or tid == tok.unk_token_id:
                tid = tok.convert_tokens_to_ids(" " + s)
            if tid is None or tid == tok.unk_token_id:
                raise ValueError(f"Token '{s}' not in vocabulary.")
            single_ids.append(tid)
            single_idx.append(i - 1)
        else:
            p, suf = s[0], s[1:]
            pid = tok.convert_tokens_to_ids(p)
            if pid is None or pid == tok.unk_token_id:
                pid = tok.convert_tokens_to_ids(" " + p)
            sid = tok.convert_tokens_to_ids(suf)
            if sid is None or sid == tok.unk_token_id:
                sid = tok.convert_tokens_to_ids(" " + suf)
            if pid is None or sid is None:
                raise ValueError(f"Multi-token '{s}' not found.")
            if pid not in multi_groups:
                multi_groups[pid] = (p, [])
            multi_groups[pid][1].append((sid, i - 1))
            suffix_ids.add(sid)

    step2 = list(suffix_ids)
    if eos is not None and eos not in step2:
        step2.append(eos)

    return {
        "single_token_ids": single_ids,
        "single_token_indices": single_idx,
        "multi_token_groups": multi_groups,
        "has_multi_token": len(multi_groups) > 0,
        "num_candidates": num_candidates,
        "eos_token_id": eos,
        "step1_token_ids": list(single_ids),
        "step2_token_ids": step2,
    }


def compute_restricted_log_probs(
    logits: Any, target_indices: list[int], device: torch.device | None = None
):
    if isinstance(logits, torch.Tensor):
        dev = device or logits.device
        tix = torch.tensor(target_indices, device=dev, dtype=torch.long)
        tgt = logits[tix]
        lse = torch.logsumexp(tgt, dim=0)
        return {int(tid): (tgt[k] - lse).item() for k, tid in enumerate(target_indices)}
    tgt = logits[target_indices]
    lse = np.log(np.sum(np.exp(tgt)))
    return {int(tid): float(tgt[k] - lse) for k, tid in enumerate(target_indices)}


def compute_joint_log_probs(
    logits_step1: Any,
    logits_step2_dict: dict[int, Any],
    candidate_info: dict[str, Any],
    step1_log_probs: dict | None = None,
) -> np.ndarray:
    n = candidate_info["num_candidates"]
    out = np.zeros(n)
    if step1_log_probs is None:
        step1_log_probs = compute_restricted_log_probs(
            logits_step1, candidate_info["step1_token_ids"]
        )
    eos = candidate_info["eos_token_id"]

    for idx, tid in zip(candidate_info["single_token_indices"], candidate_info["single_token_ids"]):
        s1 = step1_log_probs[tid]
        if tid in logits_step2_dict:
            lp2 = compute_restricted_log_probs(
                logits_step2_dict[tid],
                candidate_info["step2_token_ids"],
            )
            eos_lp = lp2.get(eos, float("-inf"))
        else:
            eos_lp = 0.0
        out[idx] = s1 + eos_lp

    for prefix_tid, (_, suf_list) in candidate_info["multi_token_groups"].items():
        pre = step1_log_probs.get(prefix_tid, float("-inf"))
        if prefix_tid not in logits_step2_dict:
            continue
        lp2 = compute_restricted_log_probs(
            logits_step2_dict[prefix_tid],
            candidate_info["step2_token_ids"],
        )
        for suf_tid, cand_idx in suf_list:
            out[cand_idx] = pre + lp2.get(suf_tid, float("-inf"))
    return out


def get_joint_log_probs(
    model: Any,
    inputs: dict[str, torch.Tensor],
    candidate_info: dict[str, Any],
    device: torch.device,
    logits_step1: torch.Tensor | None = None,
) -> np.ndarray:
    dev = device
    gen_kw = {
        "max_new_tokens": 1,
        "do_sample": False,
        "temperature": 1.0,
        "num_beams": 1,
        "output_scores": True,
        "return_dict_in_generate": True,
    }
    if logits_step1 is None:
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kw)
        logits_step1 = out.scores[0][0].cpu()
        del out
        torch.cuda.empty_cache()
    else:
        logits_step1 = (
            logits_step1.cpu() if isinstance(logits_step1, torch.Tensor) else logits_step1
        )

    if not candidate_info["has_multi_token"]:
        joint = np.zeros(candidate_info["num_candidates"])
        for idx, tid in zip(
            candidate_info["single_token_indices"], candidate_info["single_token_ids"]
        ):
            joint[idx] = float(logits_step1[tid])
        return joint

    logits_step2_dict: dict[int, torch.Tensor] = {}
    for prefix_tid in candidate_info["multi_token_groups"]:
        pt = torch.tensor([[prefix_tid]], device=dev)
        new_ids = torch.cat([inputs["input_ids"], pt], dim=1)
        new_in = {
            k: (v.clone() if isinstance(v, torch.Tensor) and k != "input_ids" else v)
            for k, v in inputs.items()
        }
        new_in["input_ids"] = new_ids
        if "attention_mask" in new_in:
            one = torch.ones((1, 1), device=dev, dtype=new_in["attention_mask"].dtype)
            new_in["attention_mask"] = torch.cat([new_in["attention_mask"], one], dim=1)
        with torch.no_grad():
            o2 = model.generate(**new_in, **gen_kw)
        logits_step2_dict[prefix_tid] = o2.scores[0][0].cpu()
        del o2, new_in, new_ids, pt
        torch.cuda.empty_cache()

    joint = compute_joint_log_probs(logits_step1, logits_step2_dict, candidate_info)
    del logits_step2_dict
    return joint


class AttentionHook:
    _warned_no_attention_global = False

    def __init__(self, target_layers: list[int]):
        self.target_layers = set(target_layers)
        self.attention_weights: dict[int, torch.Tensor] = {}
        self.handles_forward: list[Any] = []

    def _layers(self, model: Any):
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        if (
            hasattr(model, "model")
            and hasattr(model.model, "language_model")
            and hasattr(model.model.language_model, "layers")
        ):
            return model.model.language_model.layers
        if (
            hasattr(model, "language_model")
            and hasattr(model.language_model, "model")
            and hasattr(model.language_model.model, "layers")
        ):
            return model.language_model.model.layers
        raise ValueError("Cannot locate transformer layers on model.")

    def _find_attention_tensor(self, value: Any) -> torch.Tensor | None:
        if isinstance(value, torch.Tensor):
            if value.dim() == 4 and value.numel() > 0 and torch.all(value >= 0):
                return value
            return None
        if isinstance(value, (tuple, list)):
            values = value
        elif hasattr(value, "to_tuple"):
            values = value.to_tuple()
        else:
            return None

        for item in values:
            found = self._find_attention_tensor(item)
            if found is not None:
                return found
        return None

    def _hook_fn(self, layer_idx: int):
        def forward_hook(_module, _inp, output):
            if layer_idx not in self.target_layers:
                return
            attn = self._find_attention_tensor(output)
            if attn is None:
                if not AttentionHook._warned_no_attention_global:
                    print(
                        "Warning: attention hook did not capture attention weights. "
                        "The active attention backend may not return attentions."
                    )
                    AttentionHook._warned_no_attention_global = True
                return
            q_len = attn.shape[2]
            if q_len > 1:
                self.attention_weights[layer_idx] = attn.detach().clamp_min(0)

        return forward_hook

    def register_hooks(self, model: Any) -> None:
        layers = self._layers(model)
        for idx, layer in enumerate(layers):
            if idx in self.target_layers:
                h = layer.self_attn.register_forward_hook(self._hook_fn(idx))
                self.handles_forward.append(h)

    def remove_hooks(self) -> None:
        for h in self.handles_forward:
            h.remove()
        self.handles_forward.clear()

    def clear(self) -> None:
        self.attention_weights.clear()


def aggregate_layer_scores(
    attention_weights: dict[int, torch.Tensor],
    image_ranges: list[tuple[int, int]],
    query_idx: int,
    target_layers: list[int],
    num_images: int,
) -> np.ndarray:
    if not attention_weights:
        return np.zeros((len(target_layers), num_images))
    scores = np.zeros((len(target_layers), num_images))
    for li, layer_idx in enumerate(target_layers):
        tensor = attention_weights.get(layer_idx)
        if tensor is None:
            continue
        attn = tensor[0]
        qa = attn[:, query_idx, :]
        _, seq_len = qa.shape
        layer_scores = np.zeros(num_images)
        for img_idx, (start, end) in enumerate(image_ranges):
            start = max(0, min(start, seq_len))
            end = max(start, min(end, seq_len))
            if end <= start:
                continue
            sl = qa[:, start:end]
            layer_scores[img_idx] = max(0.0, sl.sum(dim=1).mean().cpu().item())
        scores[li] = np.clip(layer_scores, 0, None)
    return scores


def select_attention_scores(
    layer_scores: np.ndarray,
    bias_scores: np.ndarray | None = None,
    topk: int = 2,
) -> np.ndarray:
    topk = min(topk, layer_scores.shape[0])
    if topk == 0:
        return np.zeros(layer_scores.shape[1])
    avg = layer_scores.mean(axis=1)
    top_ix = np.argsort(avg)[-topk:]
    sel = layer_scores[top_ix].mean(axis=0)
    if bias_scores is None:
        return sel
    eps = 1e-20
    bias_avg = np.mean(np.log(np.clip(bias_scores[top_ix], 0, None) + eps), axis=0)
    return np.exp(np.log(np.clip(sel, 0, None) + eps) - bias_avg)


def compute_bias_matrix(
    stats: dict[int, list[np.ndarray]], num_candidates: int, verbose: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if verbose:
        print("\n" + "\nComputing L, I, B\n")
    logits_obs = np.zeros((num_candidates, num_candidates))
    for i in range(num_candidates):
        logits_obs[i] = np.mean(stats[i], axis=0) if stats[i] else 0.0
        if verbose and not stats[i]:
            print(f"[warn] no stats for GT row {i}")
    mu = np.mean(logits_obs)
    k_best = float("-inf")
    for i in range(num_candidates):
        for j in range(num_candidates):
            if i != j:
                k_best = max(k_best, logits_obs[i, i] - logits_obs[i, j])
    if k_best == float("-inf"):
        k_best = 0.0
    n = num_candidates
    b = mu - k_best / n
    a = b + k_best
    ideal = np.full((n, n), b, dtype=float)
    np.fill_diagonal(ideal, a)
    bias_mat = logits_obs - ideal
    if verbose:
        print("B (bias matrix) computed.")
    return bias_mat, ideal, logits_obs


def sharpen_distribution(scores: np.ndarray, power: float = 5.0) -> np.ndarray:
    eps = 1e-20
    scores = np.clip(scores, 0, None)
    x = np.log(scores + eps)
    x -= x.max()
    x *= power
    z = np.log(np.exp(x).sum())
    return np.exp(x - z)


def collect_logits_stats(
    model: Any,
    processor: Any,
    meta_data: list[dict[str, Any]],
    target_layers: list[int],
    candidate_info: dict[str, Any],
    args: Any,
    num_samples: int = 5,
) -> tuple[dict[int, list[np.ndarray]], np.ndarray, dict[str, Any]]:
    model_param = next(model.parameters())
    device = model_param.device
    dtype = model_param.dtype
    n_cand = candidate_info["num_candidates"]
    stats: dict[int, list[np.ndarray]] = {i: [] for i in range(n_cand)}
    bias_scores = np.zeros((len(target_layers), n_cand))
    total = 0
    sample_count = min(num_samples, len(meta_data))
    progress = tqdm(
        total=sample_count * n_cand,
        desc="calibration",
        unit="perm",
    )

    try:
        for sample_id in range(sample_count):
            paths = meta_data[sample_id]["image_ids"]
            caption = meta_data[sample_id]["caption"]
            gt = int(meta_data[sample_id]["index"])
            for shift in range(n_cand):
                total += 1
                perm = [(i + shift) % n_cand for i in range(n_cand)]
                perm_paths = [paths[i] for i in perm]
                perm_gt = perm.index(gt)
                pils = process_images(perm_paths)
                messages = _calibration_messages(caption, args.num_views)
                inputs = encode_hf_multimodal(
                    processor,
                    messages,
                    pils,
                    device,
                    dtype,
                    getattr(args, "max_patches", None),
                )
                ranges = find_image_token_ranges(inputs["input_ids"], processor)
                qix = inputs["input_ids"].shape[1] - 1
                hook = AttentionHook(target_layers)
                hook.register_hooks(model)
                gen_kw = {
                    "max_new_tokens": 1,
                    "do_sample": False,
                    "temperature": 1.0,
                    "num_beams": 1,
                    "output_scores": True,
                    "return_dict_in_generate": True,
                    "output_attentions": True,
                }
                inp_copy = copy.deepcopy(inputs)
                with torch.no_grad():
                    outputs = model.generate(**inp_copy, **gen_kw)
                logits_first = outputs.scores[0][0].cpu()
                bias_scores += aggregate_layer_scores(
                    hook.attention_weights,
                    ranges,
                    qix,
                    target_layers,
                    n_cand,
                )
                jlp = get_joint_log_probs(
                    model, inputs, candidate_info, device, logits_step1=logits_first
                )
                stats[perm_gt].append(jlp)
                hook.remove_hooks()
                hook.clear()
                del outputs, inp_copy, pils
                torch.cuda.empty_cache()
                progress.update(1)
    finally:
        progress.close()

    bias_scores /= max(1, total)
    return stats, bias_scores, candidate_info


def run_agd_calibration(
    model: Any, processor: Any, meta_data: list[dict], args: Any, candidate_info: dict
):
    lo, hi = args.layer_range
    layers = list(range(lo, hi + 1))
    stats, bias_scores, _ = collect_logits_stats(
        model,
        processor,
        meta_data,
        layers,
        candidate_info,
        args,
        num_samples=args.calibration_samples,
    )
    bias_mat, _, _ = compute_bias_matrix(stats, candidate_info["num_candidates"], verbose=True)
    counts = {candidate: len(values) for candidate, values in stats.items()}
    print(
        "Calibration summary: "
        f"layers={layers}, "
        f"logits_per_candidate={counts}, "
        f"bias_matrix_shape={bias_mat.shape}, "
        f"bias_scores_shape={bias_scores.shape}"
    )
    print(f"B matrix:\n{np.array2string(bias_mat, precision=4, suppress_small=True)}")
    print(f"bias scores:\n{np.array2string(bias_scores, precision=4, suppress_small=True)}")
    return bias_mat, bias_scores, layers
