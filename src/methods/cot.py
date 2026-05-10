"""Chain-of-thought decoding (long generation + digit parse)."""

from __future__ import annotations

import copy
from typing import Any

import torch

from src.methods.base import BaseMethod, _base_intro, _decode_one


class CoTMethod(BaseMethod):
    @property
    def name(self) -> str:
        return "cot"

    def build_prompt(self, caption: str) -> str:
        intro = _base_intro(self.args.num_views)
        return (
            f"{intro}\n"
            "Please think step by step. First, write your reasoning inside <think> and </think> "
            f"tags. Then, after the closing </think> tag, output only the final answer index "
            f"(a single integer from 1 to {self.args.num_views}).\n"
            f"Caption: {caption}\n"
            "Answer: <think>"
        )

    def predict(self, caption: str, pil_images: list[Any], image_paths: list[str]) -> str:
        inputs = self._encode(caption, pil_images, image_paths)
        gen_kw = {
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample": False,
            "temperature": 1.0,
            "num_beams": 1,
            "return_dict_in_generate": True,
        }
        inp = copy.deepcopy(inputs)
        with torch.no_grad():
            out = self.model.generate(**inp, **gen_kw)
        return _decode_one(self.processor, out.sequences, inputs["input_ids"].shape[1])
