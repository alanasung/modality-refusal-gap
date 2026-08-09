"""Logit-lens readout of refusal across layers and modalities."""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .activations import capture_hidden_states
from .vlm import VLMHandle


def layer_projections(
    handle: VLMHandle,
    item: dict[str, Any],
    direction: torch.Tensor,
) -> list[float]:
    """Project each layer's final-position residual onto ``direction``."""
    text = item.get("source_text") or item["text"]
    image = Image.open(item["image_path"]) if item.get("modality") == "image" else None
    states, _ = capture_hidden_states(handle, text=text, image=image)
    vals: list[float] = []
    for h in states:
        vec = h[0, -1, :].detach().cpu().float()
        vals.append(float(torch.dot(vec, direction)))
    return vals


def logit_lens_curve(
    handle: VLMHandle,
    items: list[dict[str, Any]],
    direction: torch.Tensor,
) -> dict[str, Any]:
    """Mean layer-wise projection for harmful text vs harmful image."""
    curves: dict[str, list[list[float]]] = {"text": [], "image": []}
    errors: list[str] = []
    for it in items:
        if it.get("label") != "harmful":
            continue
        try:
            curves[it["modality"]].append(layer_projections(handle, it, direction))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{it.get('item_id')}: {exc}")
    out: dict[str, Any] = {"status": "ok" if not errors else ("partial" if any(curves.values()) else "unavailable")}
    for mod, rows in curves.items():
        if not rows:
            out[mod] = [0.0] * handle.n_layers
            continue
        stacked = torch.tensor(rows)
        out[mod] = stacked.mean(0).tolist()
    out["deficit"] = [t - i for t, i in zip(out["text"], out["image"])]
    out["errors"] = errors
    return out
