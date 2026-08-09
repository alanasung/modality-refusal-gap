"""Cross-modal activation patching with sequence alignment.

Text and image forwards produce different sequence lengths because image patch
tokens are prepended. Blind fixed-position patching is unsafe. This module
aligns the *text* token span across modalities and patches only aligned
positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from .vlm import VLMHandle, encode_image, encode_text


@dataclass(frozen=True)
class Alignment:
    """Maps text-token positions into each modality's sequence."""

    text_seq_len: int
    image_prefix_len: int
    text_positions_in_text_run: tuple[int, ...]
    text_positions_in_image_run: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text_seq_len": self.text_seq_len,
            "image_prefix_len": self.image_prefix_len,
            "text_positions_in_text_run": list(self.text_positions_in_text_run),
            "text_positions_in_image_run": list(self.text_positions_in_image_run),
        }


def align_text_span(text_len: int, image_prefix_len: int) -> Alignment:
    """Align the shared text token span between text-only and image+text runs."""
    text_pos = tuple(range(text_len))
    image_pos = tuple(image_prefix_len + i for i in range(text_len))
    return Alignment(text_len, image_prefix_len, text_pos, image_pos)


def _hidden_states(
    handle: VLMHandle, item: dict[str, Any], modality: str
) -> tuple[list[torch.Tensor], int, int]:
    """Return (hidden_states, image_len, text_len)."""
    assert handle.backend == "synthetic", "patching pilot path requires SyntheticVLM or equivalent"
    if modality == "text":
        ids = encode_text(handle, item.get("source_text") or item["text"])
        out = handle.model.forward(ids, None, return_hidden_states=True)
        return out["hidden_states"], 0, out["text_len"]
    img = Image.open(item["image_path"])
    # Use source_text for the text stream so token ids match the text-only run.
    text = item.get("source_text") or item["text"]
    ids, pixels = encode_image(handle, img, text)
    # Re-encode with identical text ids as the text-only path.
    ids = encode_text(handle, text)
    out = handle.model.forward(ids, pixels, return_hidden_states=True)
    return out["hidden_states"], out["image_len"], out["text_len"]


def patch_text_into_image(
    handle: VLMHandle,
    text_item: dict[str, Any],
    image_item: dict[str, Any],
    layer: int,
) -> dict[str, Any]:
    """Replace aligned text-span activations in the image run with text-run values.

    Measures whether refusal projection is restored after the patch.
    """
    text_hs, _, text_len = _hidden_states(handle, text_item, "text")
    image_hs, image_len, img_text_len = _hidden_states(handle, image_item, "image")
    assert text_len == img_text_len, "text token lengths must match for alignment"
    align = align_text_span(text_len, image_len)

    src = text_hs[layer][0].detach().clone()
    dst = image_hs[layer][0].detach().clone()
    for t_pos, i_pos in zip(align.text_positions_in_text_run, align.text_positions_in_image_run):
        dst[i_pos] = src[t_pos]

    # Refusal proxy: cosine to planted refusal direction on final text token.
    direction = handle.model.refusal_dir.detach().cpu().float()
    before = image_hs[layer][0, align.text_positions_in_image_run[-1], :].cpu().float()
    after = dst[align.text_positions_in_image_run[-1]].cpu().float()
    text_ref = text_hs[layer][0, align.text_positions_in_text_run[-1], :].cpu().float()

    def proj(v: torch.Tensor) -> float:
        return float(torch.nn.functional.cosine_similarity(v.unsqueeze(0), direction.unsqueeze(0)))

    return {
        "layer": layer,
        "alignment": align.to_dict(),
        "projection_text": proj(text_ref),
        "projection_image_before": proj(before),
        "projection_image_after_patch": proj(after),
        "restoration": proj(after) - proj(before),
    }


def layerwise_patch_sweep(
    handle: VLMHandle,
    text_item: dict[str, Any],
    image_item: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        patch_text_into_image(handle, text_item, image_item, layer)
        for layer in range(handle.n_layers)
    ]
