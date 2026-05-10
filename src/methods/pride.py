"""PriDe: global position prior debiasing."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.agd import get_joint_log_probs
from src.methods.base import BaseMethod


class PriDeMethod(BaseMethod):
    """Subtract the column-mean of the bias matrix as a global position prior."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        args: Any,
        candidate_info: dict[str, Any],
        *,
        bias_matrix: np.ndarray,
    ):
        super().__init__(model, processor, args, candidate_info)
        self.global_bias = np.mean(bias_matrix, axis=0)

    @property
    def name(self) -> str:
        return "pride"

    def predict(self, caption: str, pil_images: list[Any], image_paths: list[str]) -> str:
        inputs = self._encode(caption, pil_images, image_paths)
        joint = get_joint_log_probs(self.model, inputs, self.candidate_info, self.device)
        final = joint - self.global_bias
        return str(int(np.argmax(final)) + 1)
