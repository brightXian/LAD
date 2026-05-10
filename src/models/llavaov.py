"""LLaVAOV loader."""

from __future__ import annotations

import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoProcessor


def load_llavaov(model_path: str) -> tuple[Any, Any]:
    path = os.path.expanduser(model_path)
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="eager",
    ).eval()
    return model, processor
