"""AGD (Attention-Guided Debiasing), same pipeline as former LAD / ``ours``."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from src.agd import (
    AttentionHook,
    aggregate_layer_scores,
    get_joint_log_probs,
    select_attention_scores,
    sharpen_distribution,
)
from src.methods.base import BaseMethod
from src.utils import find_image_token_ranges


class AGDMethod(BaseMethod):
    def __init__(
        self,
        model: Any,
        processor: Any,
        args: Any,
        candidate_info: dict[str, Any],
        *,
        bias_matrix: np.ndarray,
        bias_scores: np.ndarray,
        target_layers: list[int],
    ):
        super().__init__(model, processor, args, candidate_info)
        self.bias_matrix = bias_matrix
        self.bias_scores = bias_scores
        self.target_layers = target_layers
        self._hook: AttentionHook | None = None

    @property
    def name(self) -> str:
        return "agd"

    def setup(self) -> None:
        self._hook = AttentionHook(self.target_layers)
        self._hook.register_hooks(self.model)

    def cleanup(self) -> None:
        if self._hook is not None:
            self._hook.remove_hooks()
            self._hook.clear()
            self._hook = None

    def predict(self, caption: str, pil_images: list[Any], image_paths: list[str]) -> str:
        assert self._hook is not None
        inputs = self._encode(caption, pil_images, image_paths)
        ranges = find_image_token_ranges(inputs["input_ids"], self.processor)
        qix = inputs["input_ids"].shape[1] - 1
        gen_kw = {
            "max_new_tokens": 2,
            "do_sample": False,
            "temperature": 1.0,
            "num_beams": 1,
            "output_scores": True,
            "return_dict_in_generate": True,
            "output_attentions": True,
        }
        inp_copy = copy.deepcopy(inputs)
        with torch.no_grad():
            outputs = self.model.generate(**inp_copy, **gen_kw)
        layer_scores = aggregate_layer_scores(
            self._hook.attention_weights,
            ranges,
            qix,
            self.target_layers,
            self.candidate_info["num_candidates"],
        )
        atten = select_attention_scores(
            layer_scores,
            bias_scores=self.bias_scores,
            topk=self.args.topk,
        )
        power = getattr(self.args, "sharpen_power", 5.0)
        p_attn = sharpen_distribution(atten, power)
        bias_sample = np.dot(p_attn, self.bias_matrix)
        self._hook.clear()
        logits_step1 = outputs.scores[0][0].cpu()
        joint = get_joint_log_probs(
            self.model,
            inp_copy,
            self.candidate_info,
            self.device,
            logits_step1=logits_step1,
        )
        final = joint - bias_sample
        return str(int(np.argmax(final)) + 1)
