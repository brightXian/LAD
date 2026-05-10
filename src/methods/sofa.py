"""SoFA reduce position bias via attention mask modification."""

from __future__ import annotations

import math
import types
from typing import Any

import torch

from src.methods.base import BaseMethod, _decode_one
from src.utils import encode_hf_multimodal, find_image_token_ranges


def _build_image_token_mask(input_ids: torch.Tensor, processor: Any) -> torch.Tensor:
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for start, end in find_image_token_ranges(input_ids, processor):
        mask[:, start:end] = True
    return mask


def _apply_sofa_mask(
    mask: torch.Tensor, image_token_mask: torch.Tensor, sigma: float
) -> torch.Tensor:
    if sigma <= 0 or mask.dim() != 4:
        return mask
    out = mask.clone().to(torch.float32)
    _, _, tgt_len, src_len = out.shape
    value = math.log(sigma)
    for batch_idx in range(image_token_mask.shape[0]):
        idx = image_token_mask[batch_idx].nonzero(as_tuple=True)[0].to(out.device)
        src = idx[idx < src_len]
        tgt = idx[idx < tgt_len]
        if src.numel() == 0 or tgt.numel() == 0:
            continue
        region = out[batch_idx, :, tgt[:, None], src[None, :]]
        out[batch_idx, :, tgt[:, None], src[None, :]] = torch.where(
            region < -1.0,
            torch.full_like(region, value),
            region,
        )
    return out.to(mask.dtype)


def _get_decoder_layers(model: Any) -> list:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        return model.model.language_model.layers
    if hasattr(model, "language_model"):
        lm = model.language_model
        return lm.model.layers if hasattr(lm, "model") else lm.layers
    raise ValueError("Cannot find decoder layers for SoFA patch.")


class SoFAMethod(BaseMethod):
    def __init__(
        self,
        model: Any,
        processor: Any,
        args: Any,
        candidate_info: dict[str, Any],
        *,
        sigma: float = 0.5,
        every_n_layers: int = 2,
    ):
        super().__init__(model, processor, args, candidate_info)
        if not (0 < sigma < 1):
            raise ValueError(f"SoFA sigma must be in (0, 1), got {sigma}")
        self.sigma = sigma
        self.every_n_layers = every_n_layers
        self._layers: list = []
        self._forward_backups: list[tuple[Any, str, Any]] = []

    @property
    def name(self) -> str:
        return "sofa"

    def setup(self) -> None:
        self._layers = _get_decoder_layers(self.model)
        for idx, layer in enumerate(self._layers):
            attn = layer.self_attn
            attn._sofa_layer_idx = idx
            attn._sofa_img_mask = None
            forward_attr = "_old_forward" if hasattr(attn, "_old_forward") else "forward"
            original = getattr(attn, forward_attr)
            self._forward_backups.append((attn, forward_attr, original))

            def patched_forward(*args, _attn=attn, _original=original, **kwargs):
                attention_mask = kwargs.get("attention_mask")
                img_mask = getattr(_attn, "_sofa_img_mask", None)
                layer_idx = getattr(_attn, "_sofa_layer_idx", 0)
                if (
                    attention_mask is not None
                    and img_mask is not None
                    and layer_idx % self.every_n_layers == 0
                ):
                    kwargs["attention_mask"] = _apply_sofa_mask(
                        attention_mask, img_mask, self.sigma
                    )
                return _original(*args, **kwargs)

            if forward_attr == "forward":
                patched_forward = types.MethodType(
                    lambda module, *args, _patched=patched_forward, **kwargs: _patched(
                        *args, **kwargs
                    ),
                    attn,
                )
            setattr(attn, forward_attr, patched_forward)

    def cleanup(self) -> None:
        for attn, forward_attr, original in self._forward_backups:
            setattr(attn, forward_attr, original)
            attn._sofa_img_mask = None
        self._forward_backups.clear()

    def predict(self, caption: str, pil_images: list[Any], image_paths: list[str]) -> str:
        inputs = encode_hf_multimodal(
            self.processor,
            self.build_messages(caption),
            pil_images,
            self.device,
            self.dtype,
            getattr(self.args, "max_patches", None),
        )
        img_mask = _build_image_token_mask(inputs["input_ids"], self.processor)
        for layer in self._layers:
            layer.self_attn._sofa_img_mask = img_mask

        gen_kw = {
            "max_new_tokens": 2,
            "do_sample": False,
            "temperature": 1.0,
            "num_beams": 1,
            "return_dict_in_generate": True,
        }
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kw)
        return _decode_one(self.processor, out.sequences, inputs["input_ids"].shape[1])
