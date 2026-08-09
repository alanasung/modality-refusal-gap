"""Architecture-aware residual-stream capture for SyntheticVLM and HF VLMs.

Fails loudly when hooks cannot be attached — never invents random activations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from PIL import Image

from .vlm import VLMHandle, encode_image, encode_text

__all__ = [
    "capture_hidden_states",
    "last_token_residual",
    "attach_residual_steer",
    "attach_zero_direction",
]


def _decoder_blocks(model: Any) -> list[Any]:
    """Locate transformer blocks across common VLM naming schemes."""
    candidates = [
        getattr(model, "blocks", None),  # SyntheticVLM
        getattr(model, "layers", None),
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(model, "language_model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "language_model", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
    ]
    for c in candidates:
        if c is None:
            continue
        try:
            blocks = list(c)
        except TypeError:
            continue
        if blocks:
            return blocks
    raise RuntimeError(
        "could not locate decoder blocks for hidden-state capture; "
        "architectural measurements are unavailable for this checkpoint"
    )


def capture_hidden_states(
    handle: VLMHandle,
    *,
    text: str,
    image: Image.Image | None = None,
) -> tuple[tuple[torch.Tensor, ...], int]:
    """Return (per-layer residual at each block output, image_prefix_len)."""
    if handle.backend == "synthetic":
        if image is None:
            ids = encode_text(handle, text)
            out = handle.model.forward(ids, None, return_hidden_states=True)
            return tuple(out["hidden_states"]), 0
        ids, pixels = encode_image(handle, image, text)
        # Keep text ids identical for alignment.
        ids = encode_text(handle, text)
        out = handle.model.forward(ids, pixels, return_hidden_states=True)
        return tuple(out["hidden_states"]), int(out["image_len"])

    blocks = _decoder_blocks(handle.model)
    captured: list[torch.Tensor] = []

    def _hook(_mod: Any, _inp: Any, out: Any) -> None:
        h = out[0] if isinstance(out, tuple) else out
        captured.append(h.detach())

    handles = [b.register_forward_hook(_hook) for b in blocks]
    try:
        if image is None:
            ids = encode_text(handle, text)
            with torch.no_grad():
                handle.model(input_ids=ids)
            image_len = 0
        else:
            ids, pixels = encode_image(handle, image, text)
            with torch.no_grad():
                try:
                    handle.model(input_ids=ids, pixel_values=pixels)
                except TypeError:
                    handle.model(input_ids=ids)
            # Unknown image token count without processor metadata; report 0 and
            # require callers to treat alignment as best-effort. FakeHF adapters
            # may stamp ``image_prefix_len`` explicitly for measurable tests.
            image_len = int(getattr(handle.model, "image_prefix_len", 0) or 0)
    finally:
        for h in handles:
            h.remove()

    if not captured:
        raise RuntimeError("hooks fired no activations; cannot fabricate residuals")
    return tuple(captured), image_len


def last_token_residual(
    handle: VLMHandle,
    *,
    text: str,
    image: Image.Image | None = None,
    layer: int,
) -> torch.Tensor:
    states, _ = capture_hidden_states(handle, text=text, image=image)
    if layer < 0 or layer >= len(states):
        raise ValueError(f"layer {layer} out of range 0..{len(states)-1}")
    return states[layer][0, -1, :].detach().cpu().float()


def attach_residual_steer(
    handle: VLMHandle,
    direction: torch.Tensor,
    *,
    layer: int,
    alpha: float,
) -> Callable[[], None]:
    """Register a forward hook that adds alpha*direction at ``layer``. Returns remover."""
    if handle.backend == "synthetic":
        # Synthetic path steers via temporary refusal_dir in callers.
        return lambda: None
    blocks = _decoder_blocks(handle.model)
    if layer < 0 or layer >= len(blocks):
        raise ValueError(f"steer layer {layer} out of range")
    direction = direction.detach()

    def _hook(_mod: Any, _inp: Any, out: Any) -> Any:
        if isinstance(out, tuple):
            h = out[0]
            d = direction.to(device=h.device, dtype=h.dtype).view(1, 1, -1)
            h = h + alpha * d
            return (h,) + out[1:]
        d = direction.to(device=out.device, dtype=out.dtype).view(1, 1, -1)
        return out + alpha * d

    h = blocks[layer].register_forward_hook(_hook)
    return h.remove


def attach_zero_direction(
    handle: VLMHandle,
    direction: torch.Tensor,
    *,
    layer: int,
) -> Callable[[], None]:
    """Ablate the 1-D subspace spanned by ``direction`` at ``layer``."""
    if handle.backend == "synthetic":
        return lambda: None
    blocks = _decoder_blocks(handle.model)
    if layer < 0 or layer >= len(blocks):
        raise ValueError(f"ablate layer {layer} out of range")
    direction = torch.nn.functional.normalize(direction.detach().float(), dim=0)

    def _hook(_mod: Any, _inp: Any, out: Any) -> Any:
        h = out[0] if isinstance(out, tuple) else out
        d = direction.to(device=h.device, dtype=h.dtype)
        # Remove projection onto d: h <- h - (h·d) d
        proj = torch.einsum("bth,h->bt", h, d).unsqueeze(-1) * d.view(1, 1, -1)
        h2 = h - proj
        if isinstance(out, tuple):
            return (h2,) + out[1:]
        return h2

    h = blocks[layer].register_forward_hook(_hook)
    return h.remove
