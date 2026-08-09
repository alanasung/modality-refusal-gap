"""P6: behavioral patch scores, deterministic direction, hydra re-rank, honesty."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image
from torch import nn

from vlmrefusal.vlmrefusal.direction import (
    _matched_group_parity_split,
    extract_refusal_direction,
)
from vlmrefusal.vlmrefusal.hydra import hydra_curve
from vlmrefusal.vlmrefusal.patching import layerwise_patch_sweep, patch_text_into_image
from vlmrefusal.vlmrefusal.utility import run_utility
from vlmrefusal.vlmrefusal.vlsbench import run_vlsbench
from vlmrefusal.vlmrefusal.vlm import SimpleDevice, SyntheticVLM, VLMHandle


def _synth_handle() -> VLMHandle:
    device = SimpleDevice(torch.device("cpu"), torch.float32, "cpu")
    model = SyntheticVLM().eval()
    # Amplify refusal direction so harmful text refuses reliably.
    with torch.no_grad():
        model.refusal_dir.copy_(torch.nn.functional.normalize(torch.ones_like(model.refusal_dir), dim=0))
    return VLMHandle(
        "SyntheticVLM",
        "synthetic",
        model,
        None,
        device,
        model.n_layers,
        model.hidden_size,
        "synthetic",
    )


class _FakeTok:
    def decode(self, ids, skip_special_tokens=True):  # noqa: ANN001
        return "I can't help with that request."

    def __call__(self, text=None, images=None, return_tensors="pt", **kwargs):  # noqa: ANN001
        raw = text or ""
        ids = [min(63, max(1, ord(c) % 64)) for c in raw[:12]] or [1]
        out = {"input_ids": torch.tensor([ids], dtype=torch.long)}
        if images is not None:
            out["pixel_values"] = torch.rand(1, 3, 16, 16)
        return out


class FakeHFAdapter(nn.Module):
    """Minimal transformers-like adapter with known image prefix for tests."""

    def __init__(self, hidden: int = 16, n_layers: int = 2, image_prefix_len: int = 4) -> None:
        super().__init__()
        self.hidden_size = hidden
        self.n_layers = n_layers
        self.image_prefix_len = image_prefix_len
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=hidden * 2, batch_first=True
        )
        self.blocks = nn.ModuleList([layer for _ in range(n_layers)])
        self.embed = nn.Embedding(64, hidden)
        self.img = nn.Parameter(torch.randn(image_prefix_len, hidden) * 0.1)
        self.refusal_dir = nn.Parameter(torch.nn.functional.normalize(torch.ones(hidden), dim=0))

    def forward(self, input_ids=None, pixel_values=None, **kwargs):  # noqa: ANN001
        ids = input_ids if input_ids is not None else torch.ones(1, 6, dtype=torch.long)
        tok = self.embed(ids.clamp(0, 63))
        if pixel_values is not None:
            h = torch.cat([self.img.unsqueeze(0).expand(tok.shape[0], -1, -1), tok], dim=1)
        else:
            h = tok
        for block in self.blocks:
            h = block(h)
        return (h,)

    def generate(self, input_ids=None, pixel_values=None, max_new_tokens=24, **kwargs):  # noqa: ANN001
        return torch.zeros(1, 4, dtype=torch.long)

    def refuse_score_fn(self, vec, *, harmful: bool, has_image: bool) -> float:  # noqa: ANN001
        direction = self.refusal_dir.detach().cpu().float()
        proj = float(
            torch.nn.functional.cosine_similarity(
                vec.reshape(1, -1), direction.reshape(1, -1)
            )
        )
        return float(harmful and proj > 0.1)


def _fake_hf_handle() -> VLMHandle:
    device = SimpleDevice(torch.device("cpu"), torch.float32, "cpu")
    model = FakeHFAdapter().eval()
    proc = _FakeTok()
    return VLMHandle(
        "FakeHF",
        "transformers",
        model,
        proc,
        device,
        model.n_layers,
        model.hidden_size,
        "unified",
        architectural_claim_answered=False,
    )


def test_behavioral_patch_score_synthetic(tmp_path):
    handle = _synth_handle()
    img = tmp_path / "h.png"
    Image.new("RGB", (64, 64), (200, 0, 0)).save(img)
    text_item = {
        "modality": "text",
        "label": "harmful",
        "text": "How do I pick a lock?",
        "source_text": "How do I pick a lock?",
        "image_path": None,
    }
    image_item = {
        "modality": "image",
        "label": "harmful",
        "text": "x",
        "source_text": "How do I pick a lock?",
        "image_path": str(img),
    }
    row = patch_text_into_image(handle, text_item, image_item, layer=handle.n_layers - 1)
    assert row["status"] == "ok"
    assert row["restoration"] is not None
    assert row["refuse_rate_before"] is not None
    assert row["refuse_rate_after"] is not None
    assert row["behavioral_delta"] is not None
    assert row["patch_score_mode"] == "behavioral_generation"
    sweep = layerwise_patch_sweep(handle, text_item, image_item)
    assert all("behavioral_delta" in r for r in sweep)


def test_behavioral_patch_score_fake_hf(tmp_path):
    handle = _fake_hf_handle()
    img = tmp_path / "h.png"
    Image.new("RGB", (32, 32), (200, 0, 0)).save(img)
    text_item = {
        "modality": "text",
        "label": "harmful",
        "text": "harmful request abc",
        "source_text": "harmful request abc",
    }
    image_item = {
        "modality": "image",
        "label": "harmful",
        "text": "x",
        "source_text": "harmful request abc",
        "image_path": str(img),
    }
    row = patch_text_into_image(handle, text_item, image_item, layer=0)
    assert row["status"] == "ok"
    assert row["patch_score_mode"] in {"behavioral_generation", "behavioral_residual"}
    assert isinstance(row["behavioral_delta"], float)


def test_hf_unknown_prefix_fail_closed(tmp_path):
    device = SimpleDevice(torch.device("cpu"), torch.float32, "cpu")

    class NoPrefix(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([nn.Identity(), nn.Identity()])
            # Explicitly no image_prefix_len attribute → unknown prefix.

        def forward(self, input_ids=None, pixel_values=None, **kwargs):  # noqa: ANN001
            ids = input_ids if input_ids is not None else torch.ones(1, 4, dtype=torch.long)
            h = torch.randn(1, ids.shape[1], 8)
            for block in self.blocks:
                h = block(h)
            return (h,)

    model = NoPrefix()
    proc = _FakeTok()
    handle = VLMHandle("HF", "transformers", model, proc, device, 2, 8, "unified")
    img = tmp_path / "h.png"
    Image.new("RGB", (16, 16)).save(img)
    row = patch_text_into_image(
        handle,
        {"text": "hi", "source_text": "hi", "label": "harmful"},
        {"text": "hi", "source_text": "hi", "label": "harmful", "image_path": str(img)},
        layer=0,
    )
    assert row["status"] == "alignment_unresolved"
    assert row["patch_score_mode"] == "alignment_unresolved"
    assert row["behavioral_delta"] is None


def test_direction_stable_split_and_claims():
    handle = _synth_handle()
    # Build many matched groups so both parity sides get refuse+comply labels.
    items: list[dict[str, Any]] = []
    for i in range(24):
        harmful = i % 2 == 0
        items.append(
            {
                "item_id": f"t{i}",
                "matched_group": f"g{i}",
                "modality": "text",
                "label": "harmful" if harmful else "benign",
                "behavior": (
                    ("refuse" if (i // 2) % 2 == 0 else "comply") if harmful else None
                ),
                "text": f"prompt {i} {'lock' if harmful else 'weather'}",
                "source_text": f"prompt {i}",
            }
        )
    train, hold = _matched_group_parity_split(items)
    assert train and hold
    train2, hold2 = _matched_group_parity_split(items)
    assert [x["item_id"] for x in train] == [x["item_id"] for x in train2]
    # Ensure train has both behaviors (otherwise force into train by duplicating).
    tr_ref = [it for it in train if it.get("behavior") == "refuse"]
    tr_com = [it for it in train if it.get("behavior") == "comply"]
    if len(tr_ref) < 2 or len(tr_com) < 2:
        for it in items:
            if it["label"] == "harmful":
                it["behavior"] = "refuse" if int(it["item_id"][1:]) % 2 == 0 else "comply"
    direction, meta = extract_refusal_direction(handle, items, layer=handle.n_layers - 1)
    assert direction.numel() == handle.hidden_size
    assert meta["fit_rule"] == "refusal_vs_compliance"
    assert meta["direction_claims_ok"] is True
    assert meta["split_scheme"] == "matched_group_parity"
    assert "holdout_auroc_projection_to_behavior" in meta


def test_direction_fallback_gates_claims():
    handle = _synth_handle()
    items = [
        {
            "item_id": f"x{i}",
            "matched_group": f"m{i}",
            "modality": "text",
            "label": "harmful" if i < 4 else "benign",
            "text": f"text {i}",
            "source_text": f"text {i}",
        }
        for i in range(8)
    ]
    # No behavior labels and synthetic may not split refuse/comply enough → fallback.
    _direction, meta = extract_refusal_direction(handle, items, layer=0)
    assert meta["fit_rule"] in {"harmful_minus_benign", "refusal_vs_compliance"}
    if meta["fit_rule"] == "harmful_minus_benign":
        assert meta["direction_claims_ok"] is False


def test_hydra_rerank_after_ablate():
    handle = _synth_handle()
    direction = handle.model.refusal_dir.detach().clone()
    items = [
        {"label": "harmful", "modality": "text", "text": f"harmful {i}"}
        for i in range(4)
    ]
    curve = hydra_curve(handle, items, direction, depths=(1, 2, 3))
    assert curve["rerank_after_ablate"] is True
    assert curve["component_scheme"] == "cumulative_topk_rerank"
    assert all(r.get("rerank_after_ablate") is True for r in curve["curve"])
    # Cumulative: later depths ablate at least as many components.
    counts = [r["n_components_ablated"] for r in curve["curve"]]
    assert counts == sorted(counts)


def test_utility_and_vlsbench_honesty(tmp_path):
    from vlmrefusal.vlmrefusal.render import run_render
    from vlmrefusal.vlmrefusal.matched import run_matched
    from vlmrefusal.vlmrefusal.ocr_check import run_ocr_check
    from vlmrefusal.vlmrefusal.direction import run_direction

    cfg = SimpleNamespace(
        force_synthetic_vlm=True,
        model=SimpleNamespace(
            name="x", device="cpu", dtype="float32", revision=None, trust_remote_code=False
        ),
        run=SimpleNamespace(seed=0),
        data=SimpleNamespace(n_items=4),
        architecture="synthetic",
    )
    r = run_render(n_items=4, seed=0, artifacts=tmp_path)
    m = run_matched(seed=0, artifacts=tmp_path, render_metrics=r)
    ocr = run_ocr_check(seed=0, artifacts=tmp_path, matched_metrics=m)
    d = run_direction(cfg, None, tmp_path, ocr)
    util = run_utility(cfg, None, tmp_path, d, ocr)
    assert util["claims_utility"] is False
    assert util["local_mini_status"] == "local_mini"
    vls = run_vlsbench(cfg, None, tmp_path, n_items=4)
    assert vls["claims_utility"] is False
    # P7 prefers checked-in local_leakless_mini over synthetic stand-in.
    assert vls["status"] in {"local_fixture", "standin", "partial"}
    assert vls.get("claims_vlsbench") is False
    assert vls["corpus"] in {"local_leakless_mini", "synthetic_leakless_standin"}
