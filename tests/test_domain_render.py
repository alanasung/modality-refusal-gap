from pathlib import Path
from vlmrefusal.vlmrefusal.render import render_text_image, run_render
def test_png(tmp_path):
    p = render_text_image("hi", tmp_path/"t.png")
    assert p.exists()
def test_run(tmp_path):
    out = run_render(n_items=2, seed=0, artifacts=tmp_path)
    assert out["task"]=="render" and out["n"]==2
def test_keys(tmp_path):
    out = run_render(n_items=2, seed=0, artifacts=tmp_path)
    for k in ("task","seed","git_sha","n"): assert k in out
def test_images(tmp_path):
    import json
    out = run_render(n_items=1, seed=0, artifacts=tmp_path)
    it = json.loads(Path(out["artifact"]).read_text())["items"][0]
    assert Path(it["harmful_image"]).exists()
