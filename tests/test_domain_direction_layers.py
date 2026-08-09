from pathlib import Path
from types import SimpleNamespace
from vlmrefusal.vlmrefusal.render import run_render
from vlmrefusal.vlmrefusal.matched import run_matched
from vlmrefusal.vlmrefusal.ocr_check import run_ocr_check
from vlmrefusal.vlmrefusal.direction import run_direction
from vlmrefusal.vlmrefusal.layers import run_layers
from vlmrefusal.vlmrefusal.vlsbench import run_vlsbench
from vlmrefusal.vlmrefusal.utility import run_utility

def _ocr(tmp_path):
    r = run_render(n_items=4, seed=0, artifacts=tmp_path)
    m = run_matched(seed=0, artifacts=tmp_path, render_metrics=r)
    return run_ocr_check(seed=0, artifacts=tmp_path, matched_metrics=m)

def _cfg():
    return SimpleNamespace(force_synthetic_vlm=True, model=SimpleNamespace(name="x", device="cpu", dtype="float32", revision=None, trust_remote_code=False), run=SimpleNamespace(seed=0), data=SimpleNamespace(n_items=4), steer_alpha=1.0, architecture="synthetic")

def test_direction(tmp_path):
    out = run_direction(_cfg(), None, tmp_path, _ocr(tmp_path))
    assert Path(out["direction_path"]).exists() and "refusal_gap_text_minus_image" in out
def test_layers(tmp_path):
    ocr = _ocr(tmp_path)
    d = run_direction(_cfg(), None, tmp_path, ocr)
    out = run_layers(_cfg(), None, tmp_path, d, ocr)
    assert "best_restoration" in out
def test_vlsbench(tmp_path):
    out = run_vlsbench(_cfg(), None, tmp_path, n_items=4)
    assert out["corpus"]=="synthetic_leakless_standin"
def test_utility(tmp_path):
    ocr = _ocr(tmp_path)
    d = run_direction(_cfg(), None, tmp_path, ocr)
    out = run_utility(_cfg(), None, tmp_path, d, ocr)
    assert out["quantization"]=="none"
