import torch
from vlmrefusal.vlmrefusal.vlm import SyntheticVLM, load_vlm, is_refusal, encode_image
from types import SimpleNamespace
def test_pixels():
    m = SyntheticVLM().eval()
    out = m.forward(torch.randint(0,128,(1,10)), torch.rand(1,3,64,64), return_hidden_states=True)
    assert out["image_len"]==m.image_tokens
def test_seq_len():
    m = SyntheticVLM().eval()
    ids = torch.randint(0,128,(1,10))
    t = m.forward(ids, None, return_hidden_states=True)
    i = m.forward(ids, torch.rand(1,3,64,64), return_hidden_states=True)
    assert i["hidden_states"][-1].shape[1] > t["hidden_states"][-1].shape[1]
def test_load_synth():
    cfg = SimpleNamespace(force_synthetic_vlm=True, model=SimpleNamespace(name="x", device="cpu", dtype="float32"), architecture="synthetic")
    h = load_vlm(cfg)
    assert h.backend=="synthetic"
def test_refusal():
    assert is_refusal("I can't help with that request.")
def test_no_4bit():
    import inspect
    from vlmrefusal.vlmrefusal import vlm as m
    src = inspect.getsource(m.load_vlm)+inspect.getsource(m._load_transformers_vlm)
    assert "load_in_4bit" not in src
def test_encode(tmp_path):
    from PIL import Image
    cfg = SimpleNamespace(force_synthetic_vlm=True, model=SimpleNamespace(name="x", device="cpu", dtype="float32"))
    h = load_vlm(cfg)
    ids, pix = encode_image(h, Image.new("RGB",(64,64)), "hi")
    assert pix.shape[1]==3
