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
    # Local MMLU / MMBench proxies so the motivating dual utility axes are present
    # in the pilot schema; full profiles swap these for real eval harness scores.
    mmlu_proxy = {
        "score": util,
        "n": min(8, int(getattr(cfg.data, "n_items", 8))),
        "status": "proxy",
        "note": "Pilot proxy; full profile runs real MMLU subsets."}
    mmbench_proxy = {
        "score": util,
        "n": min(8, int(getattr(cfg.data, "n_items", 8))),
        "status": "proxy",
        "note": "Pilot proxy; full profile runs real MMBench subsets."}

    path = write_json(
        artifacts / "utility.json",
        {
            "steering": effect,
            "benign_overrefusal_cost": overrefusal,
            "utility_proxy": util,
            "mmbench": mmbench_proxy,
            "mmlu": mmlu_proxy,
            "architectural_claim_answered": handle.architectural_claim_answered,
            "architecture": handle.architecture,
            "note": (
                "Pilot fills mmbench/mmlu with local proxies so both prior work-requested "
                "utility axes are present; full profile replaces them with real scores."
            ),
            "dtype_path": str(handle.device.dtype),
            "quantization": "none (float16/float32 Apple path only)"},
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
        mmlu_proxy=mmlu_proxy["score"],
        mmbench_proxy=mmbench_proxy["score"],
        quantization="none",
    )
