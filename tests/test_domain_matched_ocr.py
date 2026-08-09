from pathlib import Path
from vlmrefusal.vlmrefusal.render import run_render
from vlmrefusal.vlmrefusal.matched import run_matched
from vlmrefusal.vlmrefusal.ocr_check import run_ocr_check
def test_cells(tmp_path):
    r = run_render(n_items=2, seed=0, artifacts=tmp_path)
    m = run_matched(seed=0, artifacts=tmp_path, render_metrics=r)
    assert m["n"]==8
def test_ocr(tmp_path):
    r = run_render(n_items=2, seed=0, artifacts=tmp_path)
    m = run_matched(seed=0, artifacts=tmp_path, render_metrics=r)
    o = run_ocr_check(seed=0, artifacts=tmp_path, matched_metrics=m)
    assert o["n_ocr_fail"]==0
def test_modalities(tmp_path):
    r = run_render(n_items=1, seed=0, artifacts=tmp_path)
    m = run_matched(seed=0, artifacts=tmp_path, render_metrics=r)
    assert set(m["modalities"])=={"text","image"}
def test_drop_missing(tmp_path):
    import json
    r = run_render(n_items=1, seed=0, artifacts=tmp_path)
    m = run_matched(seed=0, artifacts=tmp_path, render_metrics=r)
    payload = json.loads(Path(m["artifact"]).read_text())
    for it in payload["items"]:
        if it["modality"]=="image":
            Path(it["image_path"]).unlink()
    Path(m["artifact"]).write_text(json.dumps(payload))
    o = run_ocr_check(seed=0, artifacts=tmp_path, matched_metrics=m)
    assert o["n_ocr_fail"]>=1
