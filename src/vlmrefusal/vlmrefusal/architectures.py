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

__all__ = [
    "load_subject_pair",
    "contrast_summary",
    "contrast_claim_gate",
    "run_contrast_metrics",
]


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


def _arm_measured(handle: VLMHandle | None) -> bool:
    """True when an arm is a real non-synthetic load that answered its claim."""
    if handle is None:
        return False
    if handle.backend == "synthetic" or handle.architecture == "synthetic":
        return False
    return bool(handle.architectural_claim_answered)


def contrast_claim_gate(handles: dict[str, VLMHandle | None]) -> dict[str, Any]:
    """Dual-measured gate for architecture-comparison headlines.

    Headlines require both unified and modular arms to have
    ``architectural_claim_answered`` and non-synthetic modes; otherwise
    ``contrast_claim_ok=false``.
    """
    subject = handles.get("unified")
    modular = handles.get("modular")
    unified_ok = _arm_measured(subject)
    modular_ok = _arm_measured(modular)
    contrast_claim_ok = unified_ok and modular_ok
    return {
        "contrast_claim_ok": contrast_claim_ok,
        "unified_measured": unified_ok,
        "modular_measured": modular_ok,
        "unified_architectural_claim_answered": bool(
            subject and subject.architectural_claim_answered
        ),
        "modular_architectural_claim_answered": bool(
            modular and modular.architectural_claim_answered
        ),
        "unified_backend": None if subject is None else subject.backend,
        "modular_backend": None if modular is None else modular.backend,
        "honesty": (
            "OK: dual-measured unified+modular contrast"
            if contrast_claim_ok
            else (
                "UNANSWERED: architecture-comparison headlines require both "
                "unified and modular architectural_claim_answered with "
                "non-synthetic backends; contrast_claim_ok=false"
            )
        ),
    }


def contrast_summary(handles: dict[str, VLMHandle | None]) -> dict[str, Any]:
    subject = handles.get("unified")
    modular = handles.get("modular")
    gate = contrast_claim_gate(handles)
    # Legacy single-arm flag: unified subject alone answered its load claim.
    answered = bool(subject and subject.architectural_claim_answered)
    return {
        "unified_name": None if subject is None else subject.name,
        "unified_architecture": None if subject is None else subject.architecture,
        "modular_name": None if modular is None else modular.name,
        "modular_architecture": None if modular is None else modular.architecture,
        "architectural_claim_answered": answered,
        "contrast_claim_ok": gate["contrast_claim_ok"],
        "unified_measured": gate["unified_measured"],
        "modular_measured": gate["modular_measured"],
        "primary_plan": UNIFIED_PRIMARY,
        "honesty": gate["honesty"] if not gate["contrast_claim_ok"] else (
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
    seed = int(getattr(getattr(cfg, "run", cfg), "seed", 0))

    def _measure(key: str, handle: VLMHandle) -> None:
        try:
            layer = max(0, handle.n_layers - 1)
            direction, meta = extract_refusal_direction(handle, items, layer)
            gap = measure_gap(handle, items, seed=seed)
            projs = project_modality(handle, items, direction, layer)
            hydra = hydra_curve(handle, items, direction)
            per_arch[key] = {
                "model": handle.name,
                "architecture": handle.architecture,
                "backend": handle.backend,
                "architectural_claim_answered": handle.architectural_claim_answered,
                "gap": gap.get("refusal_gap_text_minus_image"),
                "gap_status": gap.get("status"),
                "gap_ci": gap.get("gap_ci"),
                "mde": gap.get("mde"),
                "power_status": gap.get("power_status"),
                "gap_claim_ok": gap.get("gap_claim_ok"),
                "projection_gap": projs.get("text", 0.0) - projs.get("image", 0.0),
                "hydra_baseline": hydra.get("baseline_refusal_rate"),
                "fit_meta": meta,
            }
        except Exception as exc:  # noqa: BLE001
            per_arch[key] = {
                "error": str(exc),
                "model": handle.name,
                "architecture": handle.architecture,
                "backend": handle.backend,
                "architectural_claim_answered": handle.architectural_claim_answered,
            }

    _measure("unified", subject)
    # Keep a lightweight stub for the dual gate after unload.
    unified_stub = VLMHandle(
        name=subject.name,
        backend=subject.backend,
        model=None,
        processor=None,
        device=subject.device,
        n_layers=subject.n_layers,
        hidden_size=subject.hidden_size,
        architecture=subject.architecture,
        role=subject.role,
        architectural_claim_answered=subject.architectural_claim_answered,
        load_notes=list(subject.load_notes),
    )
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
    summary = contrast_summary({"unified": unified_stub, "modular": modular})
    # Restore unified facts from first measurement / stub.
    summary["unified_name"] = per_arch.get("unified", {}).get("model") or unified_stub.name
    summary["unified_architecture"] = (
        per_arch.get("unified", {}).get("architecture") or unified_stub.architecture
    )
    summary["architectural_claim_answered"] = bool(
        per_arch.get("unified", {}).get("architectural_claim_answered")
    )
    gate = contrast_claim_gate({"unified": unified_stub, "modular": modular})
    summary["contrast_claim_ok"] = gate["contrast_claim_ok"]
    summary["unified_measured"] = gate["unified_measured"]
    summary["modular_measured"] = gate["modular_measured"]
    summary["honesty"] = (
        gate["honesty"]
        if not gate["contrast_claim_ok"]
        else (
            "OK: dual-measured unified+modular contrast"
            if summary["architectural_claim_answered"]
            else summary_partial["honesty"]
        )
    )
    del modular
    gc.collect()

    interaction = None
    if "unified" in per_arch and "modular" in per_arch:
        ug = per_arch["unified"].get("gap")
        mg = per_arch["modular"].get("gap")
        if ug is not None and mg is not None:
            interaction = {"gap_unified_minus_modular": ug - mg}
            # Interaction headlines inherit the dual-measured gate.
            if not summary["contrast_claim_ok"]:
                interaction["contrast_claim_ok"] = False
                interaction["note"] = (
                    "Do not report unified−modular gap interaction as settling "
                    "architecture comparison; contrast_claim_ok=false"
                )
    out = {
        **summary,
        "per_architecture": per_arch,
        "interaction": interaction,
        "load_mode": "sequential",
    }
    path = write_json(artifacts / "architectures.json", out)
    out["artifact"] = str(path)
    return out
