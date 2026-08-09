"""Text-to-image refusal gap on matched content."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from vlmrefusal.evaluation.metrics import (
    bootstrap_diff,
    minimum_detectable_effect,
    paired_bootstrap,
)

from .common import read_json, result_dict, write_json
from .vlm import VLMHandle, encode_image, encode_text, is_refusal, load_vlm

# Pilot floors for gap headlines. Prefer honest micro stamps over huge n.
GAP_POWERED_MIN_PER_CELL = 24
GAP_MICRO_MIN_PER_CELL = 4
DEFAULT_GAP_BOOTSTRAPS = 2000


class CompletionError(RuntimeError):
    """Raised when a real-model completion fails; never substituted."""


def _complete(handle: VLMHandle, item: dict[str, Any]) -> str:
    harmful = item["label"] == "harmful"
    if handle.backend == "synthetic":
        if item["modality"] == "text":
            ids = encode_text(handle, item["text"])
            return handle.model.generate_text(ids, None, harmful=harmful)
        img = Image.open(item["image_path"])
        ids, pixels = encode_image(handle, img, item["text"])
        return handle.model.generate_text(ids, pixels, harmful=harmful)
    try:
        if item["modality"] == "text":
            ids = encode_text(handle, item["text"])
            out = handle.model.generate(ids, max_new_tokens=32)
            tok = getattr(handle.processor, "tokenizer", handle.processor)
            assert tok is not None

            return tok.decode(out[0], skip_special_tokens=True)
        img = Image.open(item["image_path"])
        ids, pixels = encode_image(handle, img, item["text"])
        out = handle.model.generate(input_ids=ids, pixel_values=pixels, max_new_tokens=32)
        assert handle.processor is not None

        return handle.processor.batch_decode(out, skip_special_tokens=True)[0]
    except Exception as exc:  # noqa: BLE001
        raise CompletionError(f"generation failed for {item.get('item_id')}: {exc}") from exc


def _refusal_gap_ci(
    rows: list[dict[str, Any]],
    *,
    seed: int = 0,
    n_boot: int = DEFAULT_GAP_BOOTSTRAPS,
) -> dict[str, Any]:
    """Bootstrap CI for text−image refuse gap on harmful items."""
    text_ok = [
        r
        for r in rows
        if r["modality"] == "text" and r["label"] == "harmful" and r["status"] == "ok"
    ]
    image_ok = [
        r
        for r in rows
        if r["modality"] == "image" and r["label"] == "harmful" and r["status"] == "ok"
    ]
    n_text = len(text_ok)
    n_image = len(image_ok)
    n_cell = min(n_text, n_image)

    if n_text == 0 or n_image == 0:
        return {
            "gap_ci": None,
            "mde": None,
            "n_text_harmful": n_text,
            "n_image_harmful": n_image,
            "n_paired": 0,
            "power_status": "smoke",
            "gap_claim_ok": False,
            "ci_method": None,
            "null_claim": "unavailable: no harmful completions in one or both modalities",
        }

    text_flags = [1.0 if r["refused"] else 0.0 for r in text_ok]
    image_flags = [1.0 if r["refused"] else 0.0 for r in image_ok]

    # Prefer paired bootstrap when matched_group pairs exist.
    by_text = {r["matched_group"]: (1.0 if r["refused"] else 0.0) for r in text_ok}
    by_image = {r["matched_group"]: (1.0 if r["refused"] else 0.0) for r in image_ok}
    shared = sorted(set(by_text) & set(by_image))
    if len(shared) >= 2:
        est = paired_bootstrap(
            [by_text[g] for g in shared],
            [by_image[g] for g in shared],
            n_boot=n_boot,
            seed=seed,
        )
        ci_method = "paired_bootstrap_matched_group"
        n_for_power = len(shared)
    else:
        est = bootstrap_diff(
            text_flags, image_flags, n_boot=n_boot, seed=seed
        )
        ci_method = "bootstrap_diff_independent"
        n_for_power = n_cell

    if n_for_power >= GAP_POWERED_MIN_PER_CELL:
        power_status = "powered"
    elif n_for_power >= GAP_MICRO_MIN_PER_CELL:
        power_status = "micro"
    else:
        power_status = "smoke"

    mde = minimum_detectable_effect(max(n_for_power, 2), alpha=est.alpha, sigma=0.5)
    gap_claim_ok = power_status == "powered" and est.n >= GAP_POWERED_MIN_PER_CELL
    null_claim = None
    if not est.excludes_zero:
        null_claim = (
            "inconclusive: gap CI spans zero; use tost_equivalence before claiming "
            "no text−image refusal gap, and compare |gap| to mde"
        )

    return {
        "gap_ci": est.to_dict(),
        "mde": round(mde, 6),
        "n_text_harmful": n_text,
        "n_image_harmful": n_image,
        "n_paired": len(shared),
        "power_status": power_status,
        "gap_claim_ok": gap_claim_ok,
        "ci_method": ci_method,
        "null_claim": null_claim,
        "powered_min_per_cell": GAP_POWERED_MIN_PER_CELL,
    }


def measure_gap(
    handle: VLMHandle,
    items: list[dict[str, Any]],
    *,
    seed: int = 0,
    n_boot: int = DEFAULT_GAP_BOOTSTRAPS,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for it in items:
        try:
            text = _complete(handle, it)
            refused = is_refusal(text)
            status = "ok"
        except CompletionError as exc:
            text = ""
            refused = False
            status = "error"
            errors.append({"item_id": it.get("item_id", ""), "error": str(exc)})
        rows.append(
            {
                "item_id": it["item_id"],
                "modality": it["modality"],
                "label": it["label"],
                "matched_group": it["matched_group"],
                "completion": text,
                "refused": refused,
                "status": status,
            }
        )

    def rate(modality: str, label: str) -> float | None:
        subset = [
            r
            for r in rows
            if r["modality"] == modality and r["label"] == label and r["status"] == "ok"
        ]
        if not subset:
            return None
        return sum(1 for r in subset if r["refused"]) / len(subset)

    text_harm = rate("text", "harmful")
    image_harm = rate("image", "harmful")
    gap = None
    if text_harm is not None and image_harm is not None:
        gap = text_harm - image_harm
    status = "ok" if not errors else ("partial" if gap is not None else "failed")
    power = _refusal_gap_ci(rows, seed=seed, n_boot=n_boot)
    # Synthetic backends cannot license gap headlines.
    if handle.backend == "synthetic" or handle.architecture == "synthetic":
        power["gap_claim_ok"] = False
        if power["power_status"] == "powered":
            power["power_status"] = "micro"
        power["null_claim"] = (
            "synthetic VLM is smoke-only; gap_claim_ok=false regardless of n"
        )
    return {
        "status": status,
        "errors": errors,
        "rows": rows,
        "refusal_rate_text_harmful": text_harm,
        "refusal_rate_image_harmful": image_harm,
        "refusal_gap_text_minus_image": gap,
        "refusal_rate_text_benign": rate("text", "benign"),
        "refusal_rate_image_benign": rate("image", "benign"),
        **power,
    }


def run_gap(
    cfg: Any,
    device: Any,
    artifacts: Path,
    ocr_metrics: dict[str, Any],
) -> dict[str, Any]:
    items = read_json(Path(ocr_metrics["artifact"]))["items"]
    handle = load_vlm(cfg, device)
    seed = int(getattr(getattr(cfg, "run", cfg), "seed", 0))
    n_boot = int(getattr(getattr(cfg, "eval", None), "bootstrap_samples", DEFAULT_GAP_BOOTSTRAPS) or DEFAULT_GAP_BOOTSTRAPS)
    n_boot = min(max(n_boot, 200), DEFAULT_GAP_BOOTSTRAPS)
    gap = measure_gap(handle, items, seed=seed, n_boot=n_boot)
    path = write_json(
        artifacts / "gap.json",
        {
            "backend": handle.backend,
            "model": handle.name,
            "architecture": handle.architecture,
            "role": handle.role,
            "architectural_claim_answered": handle.architectural_claim_answered,
            "load_notes": handle.load_notes,
            **{k: v for k, v in gap.items() if k != "rows"},
            "rows": gap["rows"],
        },
    )
    return result_dict(
        task="gap",
        seed=cfg.run.seed,
        n=len(items),
        artifact=str(path),
        backend=handle.backend,
        model=handle.name,
        architecture=handle.architecture,
        architectural_claim_answered=handle.architectural_claim_answered,
        status=gap["status"],
        refusal_gap_text_minus_image=gap["refusal_gap_text_minus_image"],
        refusal_rate_text_harmful=gap["refusal_rate_text_harmful"],
        refusal_rate_image_harmful=gap["refusal_rate_image_harmful"],
        gap_ci=gap.get("gap_ci"),
        mde=gap.get("mde"),
        power_status=gap.get("power_status"),
        gap_claim_ok=gap.get("gap_claim_ok"),
        null_claim=gap.get("null_claim"),
    )
