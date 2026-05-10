"""Model loaders."""

from __future__ import annotations

from typing import Any

from src.models.internvl import load_internvl
from src.models.llavaov import load_llavaov
from src.models.qwenvl import load_qwenvl


def load_pretrained(model_path: str) -> tuple[Any, Any]:
    path = model_path.lower()
    if "qwen" in path:
        return load_qwenvl(model_path)
    if "internvl" in path:
        return load_internvl(model_path)
    if "llava" in path or "onevision" in path:
        return load_llavaov(model_path)
    raise ValueError(f"Cannot infer model type from model_path: {model_path}")
