"""Capability / over-refusal cost of any steering fix."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .common import read_json, result_dict, write_json
from .intervention import benign_overrefusal_cost, intervention_effect, utility_proxy_score
from .vlm import encode_text, is_refusal, load_vlm


def _local_mini_arithmetic(handle: Any, n: int = 8) -> dict[str, Any]:
    """Tiny local arithmetic check on SyntheticVLM / loaded models (not MMLU)."""
    ok = 0
    scored = 0
    for i in range(n):
        a, b = i, i + 1
        text = f"What is {a}+{b}? Reply with just the number."
        expect = str(a + b)
        try:
            if handle.backend == "synthetic" and hasattr(handle.model, "generate_text"):
                ids = encode_text(handle, text)
                resp = handle.model.generate_text(ids, None, harmful=False)
            else:
                ids = encode_text(handle, text)
                out = handle.model.generate(ids, max_new_tokens=8)
                tok = getattr(handle.processor, "tokenizer", handle.processor)
                assert tok is not None
                resp = tok.decode(out[0], skip_special_tokens=True)
            scored += 1
            # Synthetic stub never emits digits; count non-refusal as proxy pass.
            if expect in resp.replace(" ", ""):
                ok += 1
            elif handle.backend == "synthetic" and not is_refusal(resp):
                ok += 1
        except Exception:  # noqa: BLE001
            continue
    return {
        "score": ok / max(1, scored),
        "n": scored,
        "status": "local_mini",
        "claims_utility": False,
        "note": "Local mini arithmetic on the loaded/stub VLM; not a licensed MMLU score.",
    }


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
    local_mini = _local_mini_arithmetic(handle, n=min(8, int(getattr(cfg.data, "n_items", 8))))
    # Stand-ins must never claim licensed benchmark utility.
    mmlu_proxy = {
        "score": util,
        "n": min(8, int(getattr(cfg.data, "n_items", 8))),
        "status": "proxy",
        "claims_utility": False,
        "note": "Pilot proxy; full profile runs real MMLU subsets.",
    }
    mmbench_proxy = {
        "score": util,
        "n": min(8, int(getattr(cfg.data, "n_items", 8))),
        "status": "proxy",
        "claims_utility": False,
        "note": "Pilot proxy; full profile runs real MMBench subsets.",
    }

    path = write_json(
        artifacts / "utility.json",
        {
            "steering": effect,
            "benign_overrefusal_cost": overrefusal,
            "utility_proxy": util,
            "local_mini_arithmetic": local_mini,
            "mmbench": mmbench_proxy,
            "mmlu": mmlu_proxy,
            "claims_utility": False,
            "architectural_claim_answered": handle.architectural_claim_answered,
            "architecture": handle.architecture,
            "note": (
                "Pilot fills mmbench/mmlu with local proxies (claims_utility=false). "
                "Local mini arithmetic/OCR stand-ins use status=local_mini|proxy only."
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
        mmlu_proxy=mmlu_proxy["score"],
        mmbench_proxy=mmbench_proxy["score"],
        claims_utility=False,
        local_mini_status=local_mini["status"],
        quantization="none",
    )
