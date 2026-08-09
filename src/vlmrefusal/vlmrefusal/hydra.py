"""Ablate-and-recover sweep for redundant refusal pathways (hydra effect)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .activations import attach_zero_direction
from .common import read_json, result_dict, write_json
from .vlm import VLMHandle, encode_text, is_refusal, load_vlm

__all__ = ["run_hydra", "hydra_curve"]


@torch.no_grad()
def hydra_curve(
    handle: VLMHandle,
    items: list[dict[str, Any]],
    direction: torch.Tensor,
    *,
    depths: tuple[int, ...] = (1, 2, 3),
) -> dict[str, Any]:
    harmful = [
        it
        for it in items
        if it.get("label") == "harmful" and it.get("modality") == "text"
    ]
    if not harmful:
        harmful = items[: max(1, min(4, len(items)))]

    direction = torch.nn.functional.normalize(direction.detach().float().reshape(-1), dim=0)
    baseline = _refuse_rate(handle, harmful)
    layer = max(0, handle.n_layers - 1)
    # Rank components once: top-|coord| of the refusal direction (PCA-lite).
    ranked = torch.argsort(direction.detach().float().abs(), descending=True)
    rows = []
    for k in depths:
        kk = min(int(k), int(ranked.numel()))
        component_idx = ranked[:kk].tolist()
        if handle.backend == "synthetic":
            old = handle.model.refusal_dir.data.clone()
            ablated = old.clone().cpu().float()
            if kk > 0:
                ablated[ranked[:kk].cpu()] = 0.0
            handle.model.refusal_dir.data.copy_(
                ablated.to(device=old.device, dtype=old.dtype)
            )
            try:
                rate = _refuse_rate(handle, harmful)
            finally:
                handle.model.refusal_dir.data.copy_(old)
            component_scheme = "topk_abs_refusal_dir_coords"
        else:
            # Ablate the subspace spanned by the top-k absolute coordinates.
            partial = torch.zeros_like(direction)
            if kk > 0:
                partial[ranked[:kk]] = direction[ranked[:kk]]
            remover = attach_zero_direction(handle, partial, layer=layer)
            try:
                rate = _refuse_rate(handle, harmful)
            finally:
                remover()
            component_scheme = "topk_abs_direction_coords"
        rows.append(
            {
                "ablation_depth": k,
                "n_components_ablated": kk,
                "component_indices": component_idx[: min(8, len(component_idx))],
                "component_scheme": component_scheme,
                "refusal_rate": rate,
                "recovery_vs_baseline": rate / baseline if baseline > 0 else None,
                "architecture": handle.architecture,
                "role": handle.role,
                "layer": layer,
            }
        )
    return {
        "baseline_refusal_rate": baseline,
        "architectural_claim_answered": handle.architectural_claim_answered,
        "model": handle.name,
        "architecture": handle.architecture,
        "component_ranking": "topk_abs_direction_coords",
        "curve": rows,
    }


def _refuse_rate(handle: VLMHandle, items: list[dict[str, Any]]) -> float:
    hits = 0
    n = 0
    for it in items:
        text = it.get("text") or it.get("prompt") or "harmful request"
        n += 1
        if handle.backend == "synthetic" and hasattr(handle.model, "generate_text"):
            ids = encode_text(handle, text)
            out = handle.model.generate_text(ids, None, harmful=True)
            hits += int(is_refusal(out))
        else:
            try:
                ids = encode_text(handle, text)
                gen = handle.model.generate(ids, max_new_tokens=24)
                tok = getattr(handle.processor, "tokenizer", handle.processor)
                assert tok is not None

                out = tok.decode(gen[0], skip_special_tokens=True)
                hits += int(is_refusal(out))
            except Exception:  # noqa: BLE001
                # Do not hard-code refusal on failure.
                n -= 1
    return hits / max(1, n)


def run_hydra(
    cfg: Any,
    device: Any,
    artifacts: Path,
    direction_metrics: dict[str, Any],
    ocr_metrics: dict[str, Any],
) -> dict[str, Any]:
    items = read_json(Path(ocr_metrics["artifact"]))["items"]
    handle = load_vlm(cfg, device)
    direction_path = Path(direction_metrics["direction_path"])
    if direction_path.exists():
        blob = torch.load(direction_path, map_location="cpu")
        direction = blob["direction"]
        if blob.get("status") == "unavailable":
            return result_dict(
                task="hydra",
                seed=cfg.run.seed,
                n=len(items),
                status="unavailable",
                note="direction unavailable; hydra not run",
            )
    else:
        direction = torch.randn(handle.hidden_size)
    curve = hydra_curve(handle, items, direction)
    path = write_json(artifacts / "hydra.json", curve)
    return result_dict(
        task="hydra",
        seed=cfg.run.seed,
        n=len(items),
        artifact=str(path),
        architecture=handle.architecture,
        architectural_claim_answered=handle.architectural_claim_answered,
        baseline_refusal_rate=curve["baseline_refusal_rate"],
        status="ok",
    )
