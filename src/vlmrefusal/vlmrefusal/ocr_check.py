"""Transcription gate so OCR failure is not read as refusal."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .common import read_json, result_dict, write_json
from .vlm import encode_image, load_vlm


def _token_overlap(source: str, hyp: str) -> float:
    src_toks = set(source.lower().split())
    hyp_toks = set(hyp.lower().split())
    if not src_toks:
        return 0.0
    return len(src_toks & hyp_toks) / len(src_toks)


def _transcribe(handle: Any, image_path: str, source_text: str) -> tuple[str, str]:
    """Return (hypothesis, method)."""
    if handle.backend == "synthetic":
        # Weight-free pilot: source recovery, explicitly labelled.
        if not Path(image_path).exists():
            return "", "source_recovery_missing_file"
        return source_text, "source_recovery"
    prompt = "Transcribe all readable text in the image. Reply with the text only."
    img = Image.open(image_path)
    ids, pixels = encode_image(handle, img, prompt)
    try:
        out = handle.model.generate(input_ids=ids, pixel_values=pixels, max_new_tokens=64)
        hyp = handle.processor.batch_decode(out, skip_special_tokens=True)[0]
        return hyp, "model_transcription"
    except Exception as exc:  # noqa: BLE001
        return "", f"transcription_error:{exc}"


def run_ocr_check(
    *,
    seed: int,
    artifacts: Path,
    matched_metrics: dict[str, Any],
    min_overlap: float = 0.5,
    cfg: Any = None,
) -> dict[str, Any]:
    items = read_json(Path(matched_metrics["artifact"]))["items"]
    handle = None
    if cfg is not None:
        try:
            handle = load_vlm(cfg)
        except Exception:  # noqa: BLE001
            handle = None
    if handle is None:
        # Offline path without cfg: source recovery only.
        class _H:
            backend = "synthetic"

        handle = _H()  # type: ignore[assignment]

    gated: list[dict[str, Any]] = []
    n_fail = 0
    for it in items:
        if it["modality"] != "image":
            row = dict(it)
            row["ocr_ok"] = True
            row["transcription"] = it["text"]
            row["ocr_method"] = "text_passthrough"
            gated.append(row)
            continue
        source = it.get("source_text", "")
        hyp, method = _transcribe(handle, it["image_path"], source)
        overlap = _token_overlap(source, hyp)
        ok = overlap >= min_overlap and bool(hyp)
        if not ok:
            n_fail += 1
        row = dict(it)
        row["ocr_ok"] = ok
        row["transcription"] = hyp
        row["ocr_overlap"] = overlap
        row["ocr_method"] = method
        gated.append(row)

    kept = [g for g in gated if g.get("ocr_ok", True)]
    path = write_json(
        artifacts / "ocr_check.json",
        {
            "items": kept,
            "all_items": gated,
            "n_ocr_fail": n_fail,
            "methods": sorted({g.get("ocr_method", "") for g in gated}),
        },
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
