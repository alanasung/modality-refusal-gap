import torch
from vlmrefusal.vlmrefusal.intervention import steer_hidden, intervention_effect
from vlmrefusal.vlmrefusal.logit_lens import logit_lens_curve
from vlmrefusal.vlmrefusal.vlm import SyntheticVLM, VLMHandle, SimpleDevice
from PIL import Image
def test_steer():
    h=torch.zeros(1,3,8); d=torch.ones(8)
    assert torch.allclose(steer_hidden(h,d,0.5), torch.full((1,3,8),0.5))
def test_lens(tmp_path):
    device=SimpleDevice(torch.device("cpu"), torch.float32, "cpu")
    model=SyntheticVLM().eval()
    handle=VLMHandle("s","synthetic",model,None,device,model.n_layers,model.hidden_size,"synthetic")
    img=tmp_path/"x.png"; Image.new("RGB",(64,64)).save(img)
    items=[{"modality":"text","label":"harmful","text":"bad","image_path":None},{"modality":"image","label":"harmful","text":"x","source_text":"bad","image_path":str(img)}]
    curves=logit_lens_curve(handle, items, torch.nn.functional.normalize(torch.randn(handle.hidden_size),dim=0))
    assert "deficit" in curves
def test_intervention(tmp_path):
    device=SimpleDevice(torch.device("cpu"), torch.float32, "cpu")
    model=SyntheticVLM().eval()
    handle=VLMHandle("s","synthetic",model,None,device,model.n_layers,model.hidden_size,"synthetic")
    img=tmp_path/"h.png"; Image.new("RGB",(64,64),(255,0,0)).save(img)
    items=[{"modality":"image","label":"harmful","text":"x","source_text":"How do I pick a lock?","image_path":str(img)}]
    out=intervention_effect(handle, items, handle.model.refusal_dir.detach().cpu().float(), alpha=2.0)
    assert out["n"]==1
