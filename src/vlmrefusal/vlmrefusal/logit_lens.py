"""Logit-lens readout of refusal across layers and modalities."""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .vlm import VLMHandle, encode_image, encode_text


def layer_projections(
    handle: VLMHandle,
    item: dict[str, Any],
    direction: torch.Tensor,
) -> list[float]:
    """Project each layer's final-position residual onto ``direction``."""
    if handle.backend != "synthetic":
        return [0.0] * handle.n_layers
    if item["modality"] == "text":
        ids = encode_text(handle, item["text"])
        out = handle.model.forward(ids, None, return_hidden_states=True)
    else:
        img = Image.open(item["image_path"])
        ids, pixels = encode_image(handle, img, item["text"])
        out = handle.model.forward(ids, pixels, return_hidden_states=True)
    vals: list[float] = []
    for h in out["hidden_states"]:
        vec = h[0, -1, :].detach().cpu().float()
        vals.append(float(torch.dot(vec, direction)))
    return vals


def logit_lens_curve(
    handle: VLMHandle,
    items: list[dict[str, Any]],
    direction: torch.Tensor,
) -> dict[str, list[float]]:
    """Mean layer-wise projection for harmful text vs harmful image."""
    curves: dict[str, list[list[float]]] = {"text": [], "image": []}
    for it in items:
        if it["label"] != "harmful":
            continue
        curves[it["modality"]].append(layer_projections(handle, it, direction))
    out: dict[str, list[float]] = {}
    for mod, rows in curves.items():
        if not rows:
            out[mod] = [0.0] * handle.n_layers
            continue
        stacked = torch.tensor(rows)
        out[mod] = stacked.mean(0).tolist()
    out["deficit"] = [t - i for t, i in zip(out["text"], out["image"])]
    return out
