"""Cross-modal activation patching with sequence alignment and causal hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from .activations import capture_hidden_states
from .vlm import VLMHandle, encode_text, is_refusal


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


def _refusal_from_residual(
    handle: VLMHandle,
    residual: torch.Tensor,
    *,
    harmful: bool,
    has_image: bool,
) -> float:
    """Map a last-token residual to a refuse score in [0, 1]."""
    vec = residual.detach().cpu().float().reshape(-1)
    if handle.backend == "synthetic" and hasattr(handle.model, "refusal_dir"):
        direction = handle.model.refusal_dir.detach().cpu().float().reshape(-1)
        if direction.numel() != vec.numel():
            direction = torch.nn.functional.pad(
                direction, (0, max(0, vec.numel() - direction.numel()))
            )[: vec.numel()]
        proj = float(
            torch.nn.functional.cosine_similarity(vec.unsqueeze(0), direction.unsqueeze(0))
        )
        threshold = 0.22 if has_image else 0.15
        return float(harmful and proj > threshold)
    # FakeHF / transformers adapters may expose a refuse_score_fn.
    scorer = getattr(handle.model, "refuse_score_fn", None)
    if callable(scorer):
        return float(scorer(vec, harmful=harmful, has_image=has_image))
    direction = torch.nn.functional.normalize(vec, dim=0)
    proj = float(torch.nn.functional.cosine_similarity(vec.unsqueeze(0), direction.unsqueeze(0)))
    return float(harmful and proj > 0.5)


def _generate_refuse(
    handle: VLMHandle,
    *,
    text: str,
    image: Image.Image | None,
    harmful: bool,
) -> float:
    if handle.backend == "synthetic" and hasattr(handle.model, "generate_text"):
        if image is None:
            ids = encode_text(handle, text)
            out = handle.model.generate_text(ids, None, harmful=harmful)
        else:
            from .vlm import encode_image

            ids, pixels = encode_image(handle, image, text)
            ids = encode_text(handle, text)
            out = handle.model.generate_text(ids, pixels, harmful=harmful)
        return float(is_refusal(out))
    gen = getattr(handle.model, "generate", None)
    if callable(gen):
        try:
            ids = encode_text(handle, text)
            if image is not None and hasattr(handle.model, "forward"):
                from .vlm import encode_image

                ids, pixels = encode_image(handle, image, text)
                out_ids = gen(input_ids=ids, pixel_values=pixels, max_new_tokens=24)
            else:
                out_ids = gen(ids, max_new_tokens=24)
            tok = getattr(handle.processor, "tokenizer", handle.processor)
            if tok is None:
                return 0.0
            text_out = tok.decode(out_ids[0], skip_special_tokens=True)
            return float(is_refusal(text_out))
        except Exception:  # noqa: BLE001
            return 0.0
    return 0.0


def patch_text_into_image(
    handle: VLMHandle,
    text_item: dict[str, Any],
    image_item: dict[str, Any],
    layer: int,
) -> dict[str, Any]:
    """Patch aligned text-span activations; measure projection + behavioral delta.

    On SyntheticVLM / FakeHF this also scores refuse-rate before vs after the
    patched residual (causal within the stub). On HF models, when image
    prefix length is unknown we report status=alignment_unresolved rather than
    inventing a patch.
    """
    text = text_item.get("source_text") or text_item["text"]
    text_hs, _ = capture_hidden_states(handle, text=text, image=None)
    img = Image.open(image_item["image_path"])
    image_hs, image_len = capture_hidden_states(handle, text=text, image=img)
    # FakeHF adapters may stamp a known prefix even under transformers backend.
    if image_len == 0:
        image_len = int(getattr(handle.model, "image_prefix_len", 0) or 0)
    text_len = int(encode_text(handle, text).shape[1])
    if handle.backend != "synthetic" and image_len == 0:
        return {
            "layer": layer,
            "status": "alignment_unresolved",
            "note": "image prefix length unknown for this HF VLM; refusing unsafe blind patch",
            "refuse_rate_before": None,
            "refuse_rate_after": None,
            "behavioral_delta": None,
            "patch_score_mode": "alignment_unresolved",
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

    harmful = (text_item.get("label") == "harmful") or (image_item.get("label") == "harmful")
    refuse_before = _generate_refuse(handle, text=text, image=img, harmful=bool(harmful))
    refuse_after = _refusal_from_residual(
        handle, after, harmful=bool(harmful), has_image=True
    )
    # Also compare generation-free residual score before patch for a clean delta.
    refuse_before_resid = _refusal_from_residual(
        handle, before, harmful=bool(harmful), has_image=True
    )
    # Prefer generation baseline when available; fall back to residual-only.
    if handle.backend == "synthetic" or hasattr(handle.model, "refuse_score_fn"):
        rate_before = float(refuse_before)
        rate_after = float(refuse_after)
        mode = "behavioral_generation"
    else:
        rate_before = float(refuse_before_resid)
        rate_after = float(refuse_after)
        mode = "behavioral_residual"

    return {
        "layer": layer,
        "status": "ok",
        "alignment": align.to_dict(),
        "projection_text": proj(text_ref),
        "projection_image_before": proj(before),
        "projection_image_after_patch": proj(after),
        "restoration": proj(after) - proj(before),
        "refuse_rate_before": rate_before,
        "refuse_rate_after": rate_after,
        "behavioral_delta": rate_after - rate_before,
        "patch_score_mode": mode,
        "refuse_rate_before_residual": refuse_before_resid,
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
