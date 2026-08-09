"""Experiment stages for Refusal Without a Projector."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from omegaconf import DictConfig

from vlmrefusal.utils.io import ensure_dir
from vlmrefusal.vlmrefusal.common import cfg_n_items, cfg_seed, ensure_artifacts
from vlmrefusal.vlmrefusal.direction import run_direction
from vlmrefusal.vlmrefusal.layers import run_layers
from vlmrefusal.vlmrefusal.matched import run_matched
from vlmrefusal.vlmrefusal.ocr_check import run_ocr_check
from vlmrefusal.vlmrefusal.render import run_render
from vlmrefusal.vlmrefusal.utility import run_utility
from vlmrefusal.vlmrefusal.vlsbench import run_vlsbench


def render(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    return run_render(
        n_items=min(cfg_n_items(cfg), 16),
        seed=cfg_seed(cfg),
        artifacts=ensure_artifacts(run_dir),
    )


def matched(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    artifacts = ensure_artifacts(run_dir)
    return run_matched(
        seed=cfg_seed(cfg),
        artifacts=artifacts,
        render_metrics={"artifact": str(artifacts / "render.json")},
    )


def ocr_check(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    artifacts = ensure_artifacts(run_dir)
    return run_ocr_check(
        seed=cfg_seed(cfg),
        artifacts=artifacts,
        matched_metrics={"artifact": str(artifacts / "matched.json")},
    )


def direction(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    artifacts = ensure_artifacts(run_dir)
    return run_direction(
        cfg,
        None,
        artifacts,
        {"artifact": str(artifacts / "ocr_check.json")},
    )


def layers(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    artifacts = ensure_artifacts(run_dir)
    return run_layers(
        cfg,
        None,
        artifacts,
        {
            "direction_path": str(artifacts / "refusal_direction.pt"),
            "artifact": str(artifacts / "direction.json"),
        },
        {"artifact": str(artifacts / "ocr_check.json")},
    )


def vlsbench(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    return run_vlsbench(cfg, None, ensure_artifacts(run_dir), n_items=min(8, cfg_n_items(cfg)))


def utility(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    artifacts = ensure_artifacts(run_dir)
    return run_utility(
        cfg,
        None,
        artifacts,
        {
            "direction_path": str(artifacts / "refusal_direction.pt"),
            "artifact": str(artifacts / "direction.json"),
        },
        {"artifact": str(artifacts / "ocr_check.json")},
    )


STAGES: dict[str, Callable[[DictConfig, Path], dict[str, Any]]] = {
    "render": render,
    "matched": matched,
    "ocr_check": ocr_check,
    "direction": direction,
    "layers": layers,
    "vlsbench": vlsbench,
    "utility": utility,
}

# Standard pilot/smoke aliases.
STAGES = {
    **STAGES,
    "build_dataset": render,
    "collect": matched,
    "fit": direction,
    "evaluate": vlsbench,
    "report": utility,
}

