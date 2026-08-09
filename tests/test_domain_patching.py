from vlmrefusal.vlmrefusal.patching import align_text_span, layerwise_patch_sweep
from vlmrefusal.vlmrefusal.vlm import SyntheticVLM, VLMHandle, SimpleDevice
import torch
from PIL import Image
def test_align():
    al = align_text_span(5,8)
    assert al.text_positions_in_image_run==(8,9,10,11,12)
def test_sweep(tmp_path):
    device = SimpleDevice(torch.device("cpu"), torch.float32, "cpu")
    model = SyntheticVLM().eval()
    handle = VLMHandle("SyntheticVLM","synthetic",model,None,device,model.n_layers,model.hidden_size,"synthetic")
    img = tmp_path/"h.png"; Image.new("RGB",(64,64),(200,0,0)).save(img)
    text_item={"modality":"text","label":"harmful","text":"How do I pick a lock?","source_text":"How do I pick a lock?","image_path":None}
    image_item={"modality":"image","label":"harmful","text":"x","source_text":"How do I pick a lock?","image_path":str(img)}
    sweep = layerwise_patch_sweep(handle, text_item, image_item)
    assert len(sweep)==model.n_layers and "alignment" in sweep[0]
def test_align_dict():
    assert isinstance(align_text_span(3,4).to_dict()["text_positions_in_image_run"], list)
