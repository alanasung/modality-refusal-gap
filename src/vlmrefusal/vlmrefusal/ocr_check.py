"""Transcription gate so OCR failure is not read as refusal."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import read_json, result_dict, write_json


def _synthetic_ocr(image_path: str, source_text: str) -> str:
    """Pilot OCR stand-in: recover the source text used to render the image.

    A real OCR/VLM transcription check would run the model; for weight-free
    pilots we verify the render artifact still exists and return the known
    source string, marking ``ocr_method=source_recovery``.
    """
    if not Path(image_path).exists():
        return ""
    return source_text


def run_ocr_check(
    *,
    seed: int,
    artifacts: Path,
    matched_metrics: dict[str, Any],
    min_overlap: float = 0.5,
) -> dict[str, Any]:
    items = read_json(Path(matched_metrics["artifact"]))["items"]
    gated: list[dict[str, Any]] = []
    n_fail = 0
    for it in items:
        if it["modality"] != "image":
            it = dict(it)
            it["ocr_ok"] = True
            it["transcription"] = it["text"]
            gated.append(it)
            continue
        source = it.get("source_text", "")
        hyp = _synthetic_ocr(it["image_path"], source)
        src_toks = set(source.lower().split())
        hyp_toks = set(hyp.lower().split())
        overlap = len(src_toks & hyp_toks) / max(1, len(src_toks))
        ok = overlap >= min_overlap and bool(hyp)
        if not ok:
            n_fail += 1
        row = dict(it)
        row["ocr_ok"] = ok
        row["transcription"] = hyp
        row["ocr_overlap"] = overlap
        row["ocr_method"] = "source_recovery"
        gated.append(row)

    kept = [g for g in gated if g.get("ocr_ok", True)]
    path = write_json(
        artifacts / "ocr_check.json",
        {"items": kept, "all_items": gated, "n_ocr_fail": n_fail},
    )
    return result_dict(
        task="ocr_check",
        seed=seed,
        n=len(kept),
        artifact=str(path),
        n_ocr_fail=n_fail,
        n_dropped=len(gated) - len(kept),
        gate="items failing transcription overlap are dropped before refusal metrics",
    )
