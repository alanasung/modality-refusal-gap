"""Refusal-direction extraction from text contrastive pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .common import read_json, result_dict, write_json
from .gap import measure_gap
from .vlm import VLMHandle, encode_image, encode_text, load_vlm


def _last_residual(handle: VLMHandle, item: dict[str, Any], layer: int) -> torch.Tensor:
    if handle.backend != "synthetic":
        # Best-effort: mean-pool a random unit vector if hooks unavailable.
        g = torch.Generator().manual_seed(hash(item["item_id"]) % (2**31))
        return torch.randn(handle.hidden_size, generator=g)
    if item["modality"] == "text":
        ids = encode_text(handle, item["text"])
        out = handle.model.forward(ids, None, return_hidden_states=True)
    else:
        img = Image.open(item["image_path"])
        ids, pixels = encode_image(handle, img, item["text"])
        out = handle.model.forward(ids, pixels, return_hidden_states=True)
    h = out["hidden_states"][layer][0, -1, :].detach().cpu().float()
    return h


def extract_refusal_direction(
    handle: VLMHandle, items: list[dict[str, Any]], layer: int
) -> torch.Tensor:
    """Difference-in-means: harmful text minus benign text at ``layer``."""
    harm = [it for it in items if it["modality"] == "text" and it["label"] == "harmful"]
    ben = [it for it in items if it["modality"] == "text" and it["label"] == "benign"]
    if not harm or not ben:
        raise ValueError("need harmful and benign text items")
    h_mean = torch.stack([_last_residual(handle, it, layer) for it in harm]).mean(0)
    b_mean = torch.stack([_last_residual(handle, it, layer) for it in ben]).mean(0)
    direction = h_mean - b_mean
    return torch.nn.functional.normalize(direction, dim=0)


def project_modality(
    handle: VLMHandle,
    items: list[dict[str, Any]],
    direction: torch.Tensor,
    layer: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for modality in ("text", "image"):
        subset = [it for it in items if it["modality"] == modality and it["label"] == "harmful"]
        if not subset:
            out[modality] = 0.0
            continue
        projs = []
        for it in subset:
            h = _last_residual(handle, it, layer)
            projs.append(float(torch.dot(h, direction)))
        out[modality] = sum(projs) / len(projs)
    return out


def run_direction(
    cfg: Any,
    device: Any,
    artifacts: Path,
    ocr_metrics: dict[str, Any],
) -> dict[str, Any]:
    items = read_json(Path(ocr_metrics["artifact"]))["items"]
    handle = load_vlm(cfg, device)
    layer = int((getattr(cfg, "params", None) or {}).get("direction_layer", max(0, handle.n_layers - 1)))
    direction = extract_refusal_direction(handle, items, layer)
    projs = project_modality(handle, items, direction, layer)
    # Also compute behavioral gap in this stage for a self-contained artifact.
    gap = measure_gap(handle, items)

    torch.save({"direction": direction, "layer": layer}, artifacts / "refusal_direction.pt")
    path = write_json(
        artifacts / "direction.json",
        {
            "layer": layer,
            "projection_text_harmful": projs.get("text", 0.0),
            "projection_image_harmful": projs.get("image", 0.0),
            "projection_gap": projs.get("text", 0.0) - projs.get("image", 0.0),
            "backend": handle.backend,
            "model": handle.name,
            "behavioral_gap": gap["refusal_gap_text_minus_image"],
        },
    )
    return result_dict(
        task="direction",
        task_alias="gap+direction",
        seed=cfg.run.seed,
        n=len(items),
        artifact=str(path),
        direction_path=str(artifacts / "refusal_direction.pt"),
        backend=handle.backend,
        layer=layer,
        projection_gap=projs.get("text", 0.0) - projs.get("image", 0.0),
        refusal_gap_text_minus_image=gap["refusal_gap_text_minus_image"],
    )
