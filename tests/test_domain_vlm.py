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
    import re
    from vlmrefusal.vlmrefusal import vlm as m
    src = inspect.getsource(m.load_vlm)+inspect.getsource(m._load_transformers_vlm)
    code = re.sub(r"#.*", "", src)
    assert "load_in_4bit" not in code
    assert "BitsAndBytes" not in code
def test_encode(tmp_path):
    from PIL import Image
    cfg = SimpleNamespace(force_synthetic_vlm=True, model=SimpleNamespace(name="x", device="cpu", dtype="float32"))
    h = load_vlm(cfg)
    ids, pix = encode_image(h, Image.new("RGB",(64,64)), "hi")
    assert pix.shape[1]==3

def test_classify_unified():
    from vlmrefusal.vlmrefusal.vlm import classify_architecture, UNIFIED_PRIMARY
    assert classify_architecture(UNIFIED_PRIMARY) == "unified"
    assert classify_architecture("HuggingFaceTB/SmolVLM-500M-Instruct") == "modular"

def test_modular_refused_as_unified_subject():
    cfg = SimpleNamespace(
        force_synthetic_vlm=False,
        vlm_name="HuggingFaceTB/SmolVLM-500M-Instruct",
        architecture="modular",
        vlm_role="subject",
        allow_modular_as_subject=False,
        model=SimpleNamespace(name="HuggingFaceTB/SmolVLM-500M-Instruct", device="cpu", dtype="float32", revision=None, trust_remote_code=True),
    )
    h = load_vlm(cfg)
    assert h.backend == "synthetic"
    assert h.architectural_claim_answered is False
    assert h.role == "unanswered"

def test_force_synthetic_default_is_false_in_loader(monkeypatch):
    import vlmrefusal.vlmrefusal.vlm as vlm_mod
    calls = []
    def boom(name, cfg, info, arch):
        calls.append(name)
        raise OSError("hub unavailable in unit test")
    monkeypatch.setattr(vlm_mod, "_load_transformers_vlm", boom)
    cfg = SimpleNamespace(
        vlm_name="OpenGVLab/Mono-InternVL-2B",
        architecture="unified",
        vlm_role="subject",
        model=SimpleNamespace(name="OpenGVLab/Mono-InternVL-2B", device="cpu", dtype="float32", revision=None, trust_remote_code=True),
    )
    h = load_vlm(cfg)
    assert calls, "loader should attempt unified Hub load before synthetic fallback"
    assert h.architecture == "synthetic"
    assert h.architectural_claim_answered is False
    assert h.role == "unanswered"

def test_architectures_contrast_summary():
    from vlmrefusal.vlmrefusal.architectures import contrast_summary
    from vlmrefusal.vlmrefusal.vlm import VLMHandle, SimpleDevice
    import torch
    fake = VLMHandle(
        name="OpenGVLab/Mono-InternVL-2B",
        backend="transformers",
        model=None,
        processor=None,
        device=SimpleDevice(torch.device("cpu"), torch.float32, "cpu"),
        n_layers=1,
        hidden_size=8,
        architecture="unified",
        role="subject",
        architectural_claim_answered=True,
    )
    s = contrast_summary({"unified": fake, "modular": None})
    assert s["architectural_claim_answered"] is True
