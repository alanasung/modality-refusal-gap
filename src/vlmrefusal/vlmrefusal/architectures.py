"""Controlled unified-versus-modular contrast on identical items and metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import read_json, write_json
from .direction import extract_refusal_direction, project_modality
from .gap import measure_gap
from .hydra import hydra_curve
from .vlm import (
    MODULAR_CONTRAST,
    UNIFIED_PRIMARY,
    VLMHandle,
    classify_architecture,
    load_vlm,
)

__all__ = ["load_subject_pair", "contrast_summary", "run_contrast_metrics"]


def load_subject_pair(cfg: Any) -> dict[str, VLMHandle | None]:
    subject = load_vlm(cfg)
    force_synth = bool(getattr(cfg, "force_synthetic_vlm", False))

    class _Contrast:
        force_synthetic_vlm = force_synth
        vlm_name = str(getattr(cfg, "modular_contrast", MODULAR_CONTRAST))
        architecture = "synthetic" if force_synth else "modular"
        vlm_role = "smoke" if force_synth else "modular_baseline"
        allow_modular_as_subject = True
        model = getattr(cfg, "model", None)

    modular = load_vlm(_Contrast())
    return {"unified": subject, "modular": modular}


def contrast_summary(handles: dict[str, VLMHandle | None]) -> dict[str, Any]:
    subject = handles.get("unified")
    modular = handles.get("modular")
    answered = bool(subject and subject.architectural_claim_answered)
    return {
        "unified_name": None if subject is None else subject.name,
        "unified_architecture": None if subject is None else subject.architecture,
        "modular_name": None if modular is None else modular.name,
        "modular_architecture": None if modular is None else modular.architecture,
        "architectural_claim_answered": answered,
        "primary_plan": UNIFIED_PRIMARY,
        "honesty": (
            "OK: unified subject loaded"
            if answered
            else "UNANSWERED: do not report modular results as settling the unified question"
        ),
        "subject_manifest": None if subject is None else subject.to_manifest(),
        "modular_manifest": None if modular is None else modular.to_manifest(),
        "classified_primary": classify_architecture(UNIFIED_PRIMARY),
    }


def run_contrast_metrics(
    cfg: Any,
    artifacts: Path,
    ocr_artifact: Path,
) -> dict[str, Any]:
    """Run gap + projection + hydra sequentially on both arms (memory-safe)."""
    import gc

    items = read_json(ocr_artifact)["items"]
    # Load sequentially: subject first, then unload, then modular.
    subject = load_vlm(cfg)
    summary_partial = contrast_summary({"unified": subject, "modular": None})
    per_arch: dict[str, Any] = {}

    def _measure(key: str, handle: VLMHandle) -> None:
        try:
            layer = max(0, handle.n_layers - 1)
            direction, meta = extract_refusal_direction(handle, items, layer)
            gap = measure_gap(handle, items)
            projs = project_modality(handle, items, direction, layer)
            hydra = hydra_curve(handle, items, direction)
            per_arch[key] = {
                "model": handle.name,
                "architecture": handle.architecture,
                "architectural_claim_answered": handle.architectural_claim_answered,
                "gap": gap.get("refusal_gap_text_minus_image"),
                "gap_status": gap.get("status"),
                "projection_gap": projs.get("text", 0.0) - projs.get("image", 0.0),
                "hydra_baseline": hydra.get("baseline_refusal_rate"),
                "fit_meta": meta,
            }
        except Exception as exc:  # noqa: BLE001
            per_arch[key] = {"error": str(exc), "model": handle.name}

    _measure("unified", subject)
    del subject
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001
        pass

    force_synth = bool(getattr(cfg, "force_synthetic_vlm", False))

    class _Contrast:
        force_synthetic_vlm = force_synth
        vlm_name = str(getattr(cfg, "modular_contrast", MODULAR_CONTRAST))
        architecture = "synthetic" if force_synth else "modular"
        vlm_role = "smoke" if force_synth else "modular_baseline"
        allow_modular_as_subject = True
        model = getattr(cfg, "model", None)

    modular = load_vlm(_Contrast())
    _measure("modular", modular)
    summary = contrast_summary({"unified": None, "modular": modular})
    # Restore unified facts from first measurement.
    summary["unified_name"] = per_arch.get("unified", {}).get("model")
    summary["unified_architecture"] = per_arch.get("unified", {}).get("architecture")
    summary["architectural_claim_answered"] = bool(
        per_arch.get("unified", {}).get("architectural_claim_answered")
    )
    summary["honesty"] = (
        "OK: unified subject measured"
        if summary["architectural_claim_answered"]
        else summary_partial["honesty"]
    )
    del modular
    gc.collect()

    interaction = None
    if "unified" in per_arch and "modular" in per_arch:
        ug = per_arch["unified"].get("gap")
        mg = per_arch["modular"].get("gap")
        if ug is not None and mg is not None:
            interaction = {"gap_unified_minus_modular": ug - mg}
    out = {**summary, "per_architecture": per_arch, "interaction": interaction, "load_mode": "sequential"}
    path = write_json(artifacts / "architectures.json", out)
    out["artifact"] = str(path)
    return out
