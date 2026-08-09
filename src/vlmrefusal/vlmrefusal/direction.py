"""Refusal-direction extraction from held-out contrastive pairs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .activations import last_token_residual
from .common import read_json, result_dict, write_json
from .gap import measure_gap
from .vlm import VLMHandle, load_vlm


def _last_residual(handle: VLMHandle, item: dict[str, Any], layer: int) -> torch.Tensor:
    text = item.get("source_text") or item["text"]
    image = None
    if item.get("modality") == "image":
        image = Image.open(item["image_path"])
    return last_token_residual(handle, text=text, image=image, layer=layer)


def _stable_id(item: dict[str, Any]) -> str:
    return str(
        item.get("item_id")
        or item.get("matched_group")
        or item.get("source_text")
        or item.get("text")
        or ""
    )


def _stable_parity(item: dict[str, Any]) -> int:
    """Deterministic 0/1 split key (no Python __hash__ randomization)."""
    digest = hashlib.sha1(_stable_id(item).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2


def _matched_group_parity_split(
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split by matched_group parity so text/image twins stay on the same side."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for it in items:
        key = str(it.get("matched_group") or _stable_id(it))
        groups.setdefault(key, []).append(it)
    train: list[dict[str, Any]] = []
    hold: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items(), key=lambda kv: kv[0]):
        # Parity of the group key, not per-row hash.
        side = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % 2
        (train if side == 0 else hold).extend(rows)
    if len(train) < 2 or len(hold) < 1:
        mid = max(1, len(items) // 2)
        return items[:mid], items[mid:]
    return train, hold


def _auroc(scores: list[float], labels: list[int]) -> float | None:
    """Threshold-average AUROC; None if labels are not binary-balanced enough."""
    if len(scores) < 4 or len(set(labels)) < 2:
        return None
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum = 0.0
    for rank, (_s, y) in enumerate(pairs, start=1):
        if y == 1:
            rank_sum += rank
    # Mann–Whitney U / ROC AUC
    u = rank_sum - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _behavior_buckets(
    handle: VLMHandle, items: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Partition harmful text into refuse vs comply via model behavior when possible."""
    from .vlm import encode_text, is_refusal

    refuse: list[dict[str, Any]] = []
    comply: list[dict[str, Any]] = []
    for it in items:
        if it.get("behavior") == "refuse":
            refuse.append(it)
        elif it.get("behavior") == "comply":
            comply.append(it)
    if refuse and comply:
        return refuse, comply, "labeled_behavior"

    harm = [it for it in items if it.get("label") == "harmful" and it.get("modality") == "text"]
    for it in harm[: min(24, len(harm))]:
        try:
            text = it.get("source_text") or it["text"]
            if handle.backend == "synthetic" and hasattr(handle.model, "generate_text"):
                ids = encode_text(handle, text)
                out = handle.model.generate_text(ids, None, harmful=True)
            else:
                ids = encode_text(handle, text)
                gen = handle.model.generate(ids, max_new_tokens=24)
                tok = getattr(handle.processor, "tokenizer", handle.processor)
                assert tok is not None
                out = tok.decode(gen[0], skip_special_tokens=True)
            if is_refusal(out):
                refuse.append(it)
            else:
                comply.append(it)
        except Exception:
            continue
    if len(refuse) >= 2 and len(comply) >= 2:
        return refuse, comply, "refusal_vs_compliance"
    return [], [], "unavailable"


def extract_refusal_direction(
    handle: VLMHandle, items: list[dict[str, Any]], layer: int
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Difference-in-means on a TRAIN split; evaluate projections on HOLD-OUT.

    Prefer refusal-vs-compliance on the same harmful-intent family; fall back to
    harmful-vs-benign text with an honest fit_rule stamp. Train/hold splits use
    stable matched_group parity (never randomized ``__hash__``).
    """
    text_items = [it for it in items if it.get("modality") == "text"]
    train, hold = _matched_group_parity_split(text_items)

    refuse, comply, rule = _behavior_buckets(handle, train)
    holdout_auroc: float | None = None
    if rule != "unavailable" and refuse and comply:
        r_mean = torch.stack([_last_residual(handle, it, layer) for it in refuse]).mean(0)
        c_mean = torch.stack([_last_residual(handle, it, layer) for it in comply]).mean(0)
        direction = torch.nn.functional.normalize(r_mean - c_mean, dim=0)
        # Holdout AUROC: projection → refuse/comply when labels allow.
        hold_refuse, hold_comply, hold_rule = _behavior_buckets(handle, hold)
        if hold_rule != "unavailable" and hold_refuse and hold_comply:
            scores: list[float] = []
            labels: list[int] = []
            for it in hold_refuse:
                scores.append(float(_last_residual(handle, it, layer).dot(direction)))
                labels.append(1)
            for it in hold_comply:
                scores.append(float(_last_residual(handle, it, layer).dot(direction)))
                labels.append(0)
            holdout_auroc = _auroc(scores, labels)
        meta = {
            "n_train_refuse": len(refuse),
            "n_train_comply": len(comply),
            "n_holdout": len(hold),
            "fit_rule": "refusal_vs_compliance",
            "behavior_source": rule,
            "direction_claims_ok": True,
            "holdout_auroc_projection_to_behavior": holdout_auroc,
            "split_scheme": "matched_group_parity",
        }
        return direction, meta

    harm = [it for it in train if it.get("label") == "harmful"]
    ben = [it for it in train if it.get("label") == "benign"]
    if not harm or not ben:
        raise ValueError("need harmful and benign text items for direction fitting")
    h_mean = torch.stack([_last_residual(handle, it, layer) for it in harm]).mean(0)
    b_mean = torch.stack([_last_residual(handle, it, layer) for it in ben]).mean(0)
    direction = torch.nn.functional.normalize(h_mean - b_mean, dim=0)
    # Optional holdout AUROC against harmful/benign labels (not behavior).
    hold_scores = []
    hold_labels = []
    for it in hold:
        if it.get("label") not in {"harmful", "benign"}:
            continue
        hold_scores.append(float(_last_residual(handle, it, layer).dot(direction)))
        hold_labels.append(1 if it.get("label") == "harmful" else 0)
    holdout_auroc = _auroc(hold_scores, hold_labels)
    meta = {
        "n_train_harmful": len(harm),
        "n_train_benign": len(ben),
        "n_holdout": len(hold),
        "fit_rule": "harmful_minus_benign",
        "behavior_source": "fallback_label_contrast",
        "direction_claims_ok": False,
        "holdout_auroc_projection_to_behavior": holdout_auroc,
        "split_scheme": "matched_group_parity",
        "note": (
            "Fallback contrast; do not license as a refusal-direction claim "
            "(direction_claims_ok=false)."
        ),
    }
    return direction, meta


def project_modality(
    handle: VLMHandle,
    items: list[dict[str, Any]],
    direction: torch.Tensor,
    layer: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for modality in ("text", "image"):
        subset = [it for it in items if it.get("modality") == modality and it.get("label") == "harmful"]
        if not subset:
            out[modality] = 0.0
            continue
        projs = [_last_residual(handle, it, layer).dot(direction).item() for it in subset]
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
    try:
        direction, meta = extract_refusal_direction(handle, items, layer)
        status = "ok"
        error = None
    except Exception as exc:  # noqa: BLE001
        direction = torch.zeros(handle.hidden_size)
        meta = {"direction_claims_ok": False, "fit_rule": "unavailable"}
        status = "unavailable"
        error = str(exc)
    projs = {}
    gap: dict[str, Any] = {}
    if status == "ok":
        projs = project_modality(handle, items, direction, layer)
        gap = measure_gap(handle, items)

    torch.save(
        {
            "direction": direction,
            "layer": layer,
            "status": status,
            "fit_rule": meta.get("fit_rule"),
            "direction_claims_ok": meta.get("direction_claims_ok", False),
        },
        artifacts / "refusal_direction.pt",
    )
    path = write_json(
        artifacts / "direction.json",
        {
            "status": status,
            "error": error,
            "layer": layer,
            "fit_meta": meta,
            "fit_rule": meta.get("fit_rule"),
            "direction_claims_ok": bool(meta.get("direction_claims_ok", False)),
            "holdout_auroc_projection_to_behavior": meta.get(
                "holdout_auroc_projection_to_behavior"
            ),
            "projection_text_harmful": projs.get("text", 0.0),
            "projection_image_harmful": projs.get("image", 0.0),
            "projection_gap": projs.get("text", 0.0) - projs.get("image", 0.0),
            "backend": handle.backend,
            "model": handle.name,
            "architecture": handle.architecture,
            "architectural_claim_answered": handle.architectural_claim_answered,
            "behavioral_gap": gap.get("refusal_gap_text_minus_image"),
            "gap_status": gap.get("status"),
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
        status=status,
        fit_rule=meta.get("fit_rule"),
        direction_claims_ok=bool(meta.get("direction_claims_ok", False)),
        projection_gap=projs.get("text", 0.0) - projs.get("image", 0.0) if projs else None,
        refusal_gap_text_minus_image=gap.get("refusal_gap_text_minus_image"),
    )
