
from pathlib import Path
from types import SimpleNamespace
from vlmrefusal.vlmrefusal.render import HARMFUL_PROMPTS, BENIGN_PROMPTS, run_render
from vlmrefusal.vlmrefusal.vlm import is_refusal, SyntheticVLM, load_vlm
from vlmrefusal.vlmrefusal.patching import align_text_span
from vlmrefusal.stages import STAGES
import torch

def test_prompt_banks_nonempty():
    assert len(HARMFUL_PROMPTS)>=4 and len(BENIGN_PROMPTS)>=4

def test_is_refusal_variants():
    assert is_refusal("Sorry, I cannot do that.")
    assert not is_refusal("The capital is Paris.")

def test_align_zero_prefix():
    al = align_text_span(4, 0)
    assert al.text_positions_in_image_run == (0,1,2,3)

def test_synthetic_layers():
    assert SyntheticVLM().n_layers == 4

def test_stage_count():
    assert len(STAGES)==7

def test_force_synth_default_path():
    h = load_vlm(SimpleNamespace(force_synthetic_vlm=True, model=SimpleNamespace(name="x", device="cpu", dtype="float32")))
    assert h.architecture=="synthetic"

def test_render_n_capped(tmp_path):
    out = run_render(n_items=100, seed=0, artifacts=tmp_path)
    assert out["n"] <= len(HARMFUL_PROMPTS)*2

def test_hidden_states_len():
    m = SyntheticVLM().eval()
    out = m.forward(torch.randint(0,128,(1,5)), None, return_hidden_states=True)
    assert len(out["hidden_states"])==m.n_layers

def test_dtype_float32_cpu():
    h = load_vlm(SimpleNamespace(force_synthetic_vlm=True, model=SimpleNamespace(name="x", device="cpu", dtype="float32")))
    assert h.device.dtype == torch.float32

def test_generate_text_benign():
    m = SyntheticVLM().eval()
    text = m.generate_text(torch.randint(0,10,(1,5)), None, harmful=False)
    assert isinstance(text, str)

def test_pilot_yaml_synthetic_flag():
    text = (Path(__file__).resolve().parents[1]/"configs"/"experiment"/"pilot.yaml").read_text()
    assert "force_synthetic_vlm: true" in text
    assert "quantization: none" in text

def test_stages_order():
    assert list(STAGES.keys())[0]=="render" and list(STAGES.keys())[-1]=="utility"
