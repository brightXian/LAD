"""Instruction-style fairness prompt."""

from __future__ import annotations

from src.methods.base import BaseMethod, _base_intro


class InstructionMethod(BaseMethod):
    @property
    def name(self) -> str:
        return "instruction"

    def build_prompt(self, caption: str) -> str:
        intro = _base_intro(self.args.num_views)
        extra = (
            "Please note that the provided images have been randomly shuffled, so it is essential "
            "to consider them fairly and without bias. Analyze the visual content objectively and "
            "do not be influenced by the order of the images."
        )
        return f"{intro} {extra} Respond with the index number only and nothing else.\nCaption: {caption}\nAnswer: "
