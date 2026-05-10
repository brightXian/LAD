"""QwenVL loader."""

from __future__ import annotations

import os
from typing import Any

from transformers import (
    AutoProcessor,
)
from transformers import (
    Qwen2_5_VLForConditionalGeneration as QwenvlForConditionalGeneration,
)


def load_qwenvl(model_path: str) -> tuple[Any, Any]:
    path = os.path.expanduser(model_path)
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, padding_side="left")
    model = QwenvlForConditionalGeneration.from_pretrained(
        path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="eager",
    ).eval()
    return model, processor
