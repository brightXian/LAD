"""InternVL3-8B-HF loader."""

from __future__ import annotations

import os
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def load_internvl(model_path: str) -> tuple[Any, Any]:
    path = os.path.expanduser(model_path)
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, padding_side="left")
    model = AutoModelForImageTextToText.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="eager",
    ).eval()
    return model, processor
