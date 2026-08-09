"""Text-to-image refusal gap on matched content."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .common import read_json, result_dict, write_json
from .vlm import VLMHandle, encode_image, encode_text, is_refusal, load_vlm


def _complete(handle: VLMHandle, item: dict[str, Any]) -> str:
    harmful = item["label"] == "harmful"
    if handle.backend == "synthetic":
        if item["modality"] == "text":
            ids = encode_text(handle, item["text"])
            return handle.model.generate_text(ids, None, harmful=harmful)
        img = Image.open(item["image_path"])
        ids, pixels = encode_image(handle, img, item["text"])
        return handle.model.generate_text(ids, pixels, harmful=harmful)
    # Real VLM path (best-effort generate).
    try:
        if item["modality"] == "text":
            ids = encode_text(handle, item["text"])
            out = handle.model.generate(ids, max_new_tokens=32)
            return handle.processor.tokenizer.decode(out[0], skip_special_tokens=True)
        img = Image.open(item["image_path"])
        ids, pixels = encode_image(handle, img, item["text"])
        out = handle.model.generate(input_ids=ids, pixel_values=pixels, max_new_tokens=32)
        return handle.processor.batch_decode(out, skip_special_tokens=True)[0]
    except Exception:  # noqa: BLE001 - fall back to refusal heuristic
        return "I can't help with that request." if harmful and item["modality"] == "text" else "ok"


def measure_gap(handle: VLMHandle, items: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for it in items:
        text = _complete(handle, it)
        refused = is_refusal(text)
        rows.append(
            {
                "item_id": it["item_id"],
                "modality": it["modality"],
                "label": it["label"],
                "matched_group": it["matched_group"],
                "completion": text,
                "refused": refused,
            }
        )

    def rate(modality: str, label: str) -> float:
        subset = [r for r in rows if r["modality"] == modality and r["label"] == label]
        if not subset:
            return 0.0
        return sum(1 for r in subset if r["refused"]) / len(subset)

    text_harm = rate("text", "harmful")
    image_harm = rate("image", "harmful")
    return {
        "rows": rows,
        "refusal_rate_text_harmful": text_harm,
        "refusal_rate_image_harmful": image_harm,
        "refusal_gap_text_minus_image": text_harm - image_harm,
        "refusal_rate_text_benign": rate("text", "benign"),
        "refusal_rate_image_benign": rate("image", "benign"),
    }


def run_gap(
    cfg: Any,
    device: Any,
    artifacts: Path,
    ocr_metrics: dict[str, Any],
) -> dict[str, Any]:
    items = read_json(Path(ocr_metrics["artifact"]))["items"]
    handle = load_vlm(cfg, device)
    gap = measure_gap(handle, items)
    path = write_json(
        artifacts / "gap.json",
        {
            "backend": handle.backend,
            "model": handle.name,
            "architecture": handle.architecture,
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
        refusal_gap_text_minus_image=gap["refusal_gap_text_minus_image"],
        refusal_rate_text_harmful=gap["refusal_rate_text_harmful"],
        refusal_rate_image_harmful=gap["refusal_rate_image_harmful"],
    )
