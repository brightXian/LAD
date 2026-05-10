"""Evaluation methods."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.agd import run_agd_calibration
from src.methods.agd import AGDMethod
from src.methods.base import BaseMethod, VanillaMethod
from src.methods.cot import CoTMethod
from src.methods.fn import FNMethod
from src.methods.instruction import InstructionMethod
from src.methods.pride import PriDeMethod
from src.methods.self_consistency import SelfConsistencyMethod
from src.methods.sofa import SoFAMethod

__all__ = ["build_method", "run_agd_calibration"]


def build_method(
    name: str,
    model: Any,
    processor: Any,
    args: Any,
    candidate_info: dict[str, Any],
    *,
    bias_matrix: np.ndarray | None = None,
    bias_scores: np.ndarray | None = None,
    target_layers: list[int] | None = None,
) -> BaseMethod:
    key = name.lower()
    if key in ("base", "vanilla"):
        return VanillaMethod(model, processor, args, candidate_info)
    if key == "instruction":
        return InstructionMethod(model, processor, args, candidate_info)
    if key == "cot":
        return CoTMethod(model, processor, args, candidate_info)
    if key in ("self_consistency", "sc"):
        return SelfConsistencyMethod(model, processor, args, candidate_info)
    if key in ("agd", "lad", "ours"):
        if bias_matrix is None or bias_scores is None or target_layers is None:
            raise ValueError("AGD requires bias_matrix, bias_scores, and target_layers.")
        return AGDMethod(
            model,
            processor,
            args,
            candidate_info,
            bias_matrix=bias_matrix,
            bias_scores=bias_scores,
            target_layers=target_layers,
        )
    if key == "pride":
        if bias_matrix is None:
            raise ValueError("PriDe requires bias_matrix.")
        return PriDeMethod(model, processor, args, candidate_info, bias_matrix=bias_matrix)
    if key == "sofa":
        sigma = getattr(args, "sofa_sigma", 0.5)
        every_n = getattr(args, "sofa_every_n_layers", 2)
        return SoFAMethod(
            model, processor, args, candidate_info, sigma=sigma, every_n_layers=every_n
        )
    if key == "fn":
        k = getattr(args, "perm_avg_k", 16)
        return FNMethod(model, processor, args, candidate_info, k=k)
    raise ValueError(f"Unknown method: {name}")
