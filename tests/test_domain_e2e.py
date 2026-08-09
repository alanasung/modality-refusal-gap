from pathlib import Path
from omegaconf import OmegaConf
from vlmrefusal.stages import STAGES
def test_e2e(tmp_path):
    cfg = OmegaConf.create({
        "run":{"seed":0},
        "data":{"n_items":4},
        "model":{"name":"HuggingFaceTB/SmolVLM-500M-Instruct","device":"cpu","dtype":"float32","revision":None,"trust_remote_code":False},
        "force_synthetic_vlm": True,
        "architecture":"synthetic",
        "steer_alpha":1.0,
    })
    for name in STAGES:
        out = STAGES[name](cfg, tmp_path)
        assert out["task"]==name
    assert (tmp_path/"artifacts"/"refusal_direction.pt").exists()
