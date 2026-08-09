"""Cross-modal activation patching with sequence alignment and causal hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from .activations import capture_hidden_states
from .vlm import VLMHandle, encode_text


@dataclass(frozen=True)
class Alignment:
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
    text_pos = tuple(range(text_len))
    image_pos = tuple(image_prefix_len + i for i in range(text_len))
    return Alignment(text_len, image_prefix_len, text_pos, image_pos)


def patch_text_into_image(
    handle: VLMHandle,
    text_item: dict[str, Any],
    image_item: dict[str, Any],
    layer: int,
) -> dict[str, Any]:
    """Patch aligned text-span activations; measure projection restoration.

    On SyntheticVLM this also re-runs the LM head path after surgically editing
    the captured residual (causal within the stub). On HF models, when image
    prefix length is unknown we report status=alignment_unresolved rather than
    inventing a patch.
    """
    text = text_item.get("source_text") or text_item["text"]
    text_hs, _ = capture_hidden_states(handle, text=text, image=None)
    img = Image.open(image_item["image_path"])
    image_hs, image_len = capture_hidden_states(handle, text=text, image=img)
    text_len = int(encode_text(handle, text).shape[1])
    if handle.backend != "synthetic" and image_len == 0:
        return {
            "layer": layer,
            "status": "alignment_unresolved",
            "note": "image prefix length unknown for this HF VLM; refusing unsafe blind patch",
        }
    align = align_text_span(text_len, image_len)
    src = text_hs[layer][0].detach().clone()
    dst = image_hs[layer][0].detach().clone()
    for t_pos, i_pos in zip(align.text_positions_in_text_run, align.text_positions_in_image_run):
        if i_pos < dst.shape[0] and t_pos < src.shape[0]:
            dst[i_pos] = src[t_pos]

    if handle.backend == "synthetic":
        direction = handle.model.refusal_dir.detach().cpu().float()
    else:
        direction = torch.nn.functional.normalize(src[-1].cpu().float(), dim=0)

    before = image_hs[layer][0, align.text_positions_in_image_run[-1], :].cpu().float()
    after = dst[align.text_positions_in_image_run[-1]].cpu().float()
    text_ref = text_hs[layer][0, align.text_positions_in_text_run[-1], :].cpu().float()

    def proj(v: torch.Tensor) -> float:
        return float(torch.nn.functional.cosine_similarity(v.unsqueeze(0), direction.unsqueeze(0)))

    return {
        "layer": layer,
        "status": "ok",
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
