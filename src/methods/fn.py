"""FN (Permutation Averaging): average logits over K random permutations to cancel position bias."""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch

from src.agd import get_joint_log_probs
from src.methods.base import BaseMethod
from src.utils import encode_hf_multimodal


class FNMethod(BaseMethod):
    def __init__(
        self,
        model: Any,
        processor: Any,
        args: Any,
        candidate_info: dict[str, Any],
        *,
        k: int = 16,
    ):
        super().__init__(model, processor, args, candidate_info)
        self.k = k

    @property
    def name(self) -> str:
        return "fn"

    def predict(self, caption: str, pil_images: list[Any], image_paths: list[str]) -> str:
        n = self.args.num_views
        messages = self.build_messages(caption)

        k = min(self.k, math.factorial(n))
        base = list(range(n))
        perms: list[list[int]] = [base]
        seen = {tuple(base)}
        while len(perms) < k:
            p = tuple(random.sample(base, n))
            if p not in seen:
                seen.add(p)
                perms.append(list(p))

        accumulated = np.zeros(n)
        for perm in perms:
            perm_images = [pil_images[i] for i in perm]
            inputs = encode_hf_multimodal(
                self.processor,
                messages,
                perm_images,
                self.device,
                self.dtype,
                getattr(self.args, "max_patches", None),
            )
            log_probs = get_joint_log_probs(self.model, inputs, self.candidate_info, self.device)
            # restore to the order of pil_images: position j in perm corresponds to pil_images[perm[j]]
            restored = np.zeros(n)
            for j, orig_idx in enumerate(perm):
                restored[orig_idx] = log_probs[j]
            accumulated += restored
            del inputs
            torch.cuda.empty_cache()

        avg = accumulated / len(perms)
        return str(int(np.argmax(avg)) + 1)
