"""Capability / over-refusal cost of any steering fix."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .common import read_json, result_dict, write_json
from .intervention import benign_overrefusal_cost, intervention_effect, utility_proxy_score
from .vlm import load_vlm


def run_utility(
    cfg: Any,
    device: Any,
    artifacts: Path,
    direction_metrics: dict[str, Any],
    ocr_metrics: dict[str, Any],
) -> dict[str, Any]:
    items = read_json(Path(ocr_metrics["artifact"]))["items"]
    handle = load_vlm(cfg, device)
    direction = torch.load(direction_metrics["direction_path"], map_location="cpu")["direction"]
    alpha = float((getattr(cfg, "params", None) or {}).get("steer_alpha", 1.0))

    effect = intervention_effect(handle, items, direction, alpha=alpha)
    overrefusal = benign_overrefusal_cost(handle, items, direction, alpha=alpha)
    util = utility_proxy_score(handle, n=min(8, cfg.data.n_items))

    path = write_json(
        artifacts / "utility.json",
        {
            "steering": effect,
            "benign_overrefusal_cost": overrefusal,
            "utility_proxy": util,
            "mmbench": None,
            "mmlu": None,
            "note": (
                "MMBench/MMLU are declared for the full profile; the pilot reports "
                "a local utility proxy and benign over-refusal under steering."
            ),
            "dtype_path": str(handle.device.dtype),
            "quantization": "none (float16/float32 Apple path only)",
        },
    )
    return result_dict(
        task="utility",
        seed=cfg.run.seed,
        n=len(items),
        artifact=str(path),
        backend=handle.backend,
        steering_delta=effect.get("delta"),
        benign_overrefusal_cost=overrefusal,
        utility_proxy=util,
        quantization="none",
    )
