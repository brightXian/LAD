"""Self-consistency: multiple samples + majority vote."""

from __future__ import annotations

import copy
import re
from collections import Counter
from typing import Any

import torch

from src.methods.base import BaseMethod, _base_intro, _decode_one


class SelfConsistencyMethod(BaseMethod):
    @property
    def name(self) -> str:
        return "self_consistency"

    def build_prompt(self, caption: str) -> str:
        intro = _base_intro(self.args.num_views)
        return (
            f"{intro}\n"
            "Please think step by step to analyze the content of the images. "
            "After your reasoning, provide the final answer in the exact format: "
            '"The answer is <index>".\n'
            f"Caption: {caption}\n"
            "Answer: "
        )

    def predict(self, caption: str, pil_images: list[Any], image_paths: list[str]) -> str:
        inputs = self._encode(caption, pil_images, image_paths)
        nv = self.args.num_views
        tok = getattr(self.processor, "tokenizer", self.processor)
        gen_kw = {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.7,
            "top_k": 40,
            "num_beams": 1,
            "return_dict_in_generate": True,
            "eos_token_id": tok.eos_token_id,
            "pad_token_id": tok.pad_token_id,
        }
        votes: list[int] = []
        texts: list[str] = []
        for sample_idx in range(self.args.sc_samples):
            inp = copy.deepcopy(inputs)
            with torch.no_grad():
                out = self.model.generate(**inp, **gen_kw)
            text = _decode_one(self.processor, out.sequences, inputs["input_ids"].shape[1])
            texts.append(f"sample {sample_idx + 1}: {text}")
            m = re.search(r"[Tt]he answer is (\d+)", text)
            if m:
                k = int(m.group(1))
            else:
                nums = [int(x) for x in re.findall(r"\d+", text)]
                valid = [n for n in nums if 1 <= n <= nv]
                k = valid[-1] if valid else None
            if k is not None and 1 <= k <= nv:
                votes.append(k)
        answer = Counter(votes).most_common(1)[0][0] if votes else 1
        texts.append(f"The answer is {answer}")
        return "\n".join(texts)
