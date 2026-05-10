"""Base decoding method (vanilla index prediction)."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any

import torch

from src.utils import encode_hf_multimodal


def _base_intro(num_views: int) -> str:
    return (
        f"Given {num_views} images indexed from 1 to {num_views}, "
        "identify the image that best matches the provided caption."
    )


def _decode_one(processor: Any, sequences: torch.Tensor, offset: int) -> str:
    gen_ids = sequences[:, offset:]
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    tok = getattr(processor, "tokenizer", processor)
    return tok.decode(gen_ids[0], skip_special_tokens=True).strip()


class BaseMethod(ABC):
    def __init__(self, model: Any, processor: Any, args: Any, candidate_info: dict[str, Any]):
        self.model = model
        self.processor = processor
        self.args = args
        self.candidate_info = candidate_info
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def build_prompt(self, caption: str) -> str:
        intro = _base_intro(self.args.num_views)
        return (
            f"{intro} Respond with the index number only and nothing else.\n"
            f"Caption: {caption}\n"
            "Answer: "
        )

    def build_messages(self, caption: str) -> list[dict[str, Any]]:
        prompt = self.build_prompt(caption)
        return [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in range(self.args.num_views)],
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def setup(self) -> None:  # noqa: B027
        pass

    def cleanup(self) -> None:  # noqa: B027
        pass

    def _encode(
        self, caption: str, pil_images: list[Any], image_paths: list[str]
    ) -> dict[str, torch.Tensor]:
        messages = self.build_messages(caption)
        return encode_hf_multimodal(
            self.processor,
            messages,
            pil_images,
            self.device,
            self.dtype,
            getattr(self.args, "max_patches", None),
        )

    def predict(self, caption: str, pil_images: list[Any], image_paths: list[str]) -> str:
        inputs = self._encode(caption, pil_images, image_paths)
        gen_kw = {
            "max_new_tokens": 2,
            "do_sample": False,
            "temperature": 1.0,
            "num_beams": 1,
            "return_dict_in_generate": True,
        }
        inp = copy.deepcopy(inputs)
        with torch.no_grad():
            out = self.model.generate(**inp, **gen_kw)
        return _decode_one(self.processor, out.sequences, inputs["input_ids"].shape[1])


class VanillaMethod(BaseMethod):
    @property
    def name(self) -> str:
        return "base"
