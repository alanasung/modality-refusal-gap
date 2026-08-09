"""Layer-wise projection (logit lens) and cross-modal patching stage body."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .common import read_json, result_dict, write_json
from .logit_lens import logit_lens_curve
from .patching import layerwise_patch_sweep
from .vlm import load_vlm


def run_layers(
    cfg: Any,
    device: Any,
    artifacts: Path,
    direction_metrics: dict[str, Any],
    ocr_metrics: dict[str, Any],
) -> dict[str, Any]:
    items = read_json(Path(ocr_metrics["artifact"]))["items"]
    handle = load_vlm(cfg, device)
    direction = torch.load(direction_metrics["direction_path"], map_location="cpu")["direction"]

    curves = logit_lens_curve(handle, items, direction)

    # Pick one matched harmful pair for the patching sweep.
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for it in items:
        if it["label"] != "harmful":
            continue
        groups.setdefault(it["matched_group"], {})[it["modality"]] = it
    pair = next(g for g in groups.values() if "text" in g and "image" in g)

    try:
        sweep = layerwise_patch_sweep(handle, pair["text"], pair["image"])
        patch_status = "ok"
    except Exception as exc:  # noqa: BLE001
        sweep = [
            {
                "layer": i,
                "restoration": None,
                "status": "unavailable",
                "error": str(exc),
            }
            for i in range(handle.n_layers)
        ]
        patch_status = "unavailable"

    scored = [r for r in sweep if r.get("restoration") is not None]
    best = max(scored, key=lambda r: r.get("restoration", 0.0)) if scored else {"layer": None, "restoration": None}
    path = write_json(
        artifacts / "layers.json",
        {
            "logit_lens": curves,
            "patch_sweep": sweep,
            "patch_status": patch_status,
            "best_patch_layer": best.get("layer"),
            "best_restoration": best.get("restoration"),
            "backend": handle.backend,
            "architecture": handle.architecture,
            "architectural_claim_answered": handle.architectural_claim_answered,
            "alignment_note": (
                "Cross-modal patching aligns the shared text token span; image "
                "prefix tokens are never overwritten by text residuals. "
                "HF models with unknown image-prefix length return "
                "status=alignment_unresolved rather than inventing zeros."
            ),
        },
    )
    return result_dict(
        task="layers",
        seed=cfg.run.seed,
        n=len(items),
        artifact=str(path),
        backend=handle.backend,
        best_patch_layer=best.get("layer"),
        best_restoration=(
            None if best.get("restoration") is None else float(best.get("restoration") or 0)
        ),
        patch_status=patch_status,
        mean_deficit=float(sum(curves["deficit"]) / max(1, len(curves["deficit"]))),
    )
