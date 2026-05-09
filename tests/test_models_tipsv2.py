import torch
import timm

def test_tipsv2_b14_forward_shapes():
    model = timm.create_model("tipsv2_b14", pretrained=False)
    model.eval()
    x = torch.rand(1, 3, 448, 448)
    with torch.no_grad():
        cls, regs, patches = model(x)
    assert cls.shape == (1, 1, 768)
    assert regs.shape == (1, 1, 768)
    assert patches.shape[0] == 1
    assert patches.shape[-1] == 768
    assert patches.shape[1] == 1024
    assert patches.shape[2] == 768

def test_tipsv2_l14_forward_shapes():
    model = timm.create_model("tipsv2_l14", pretrained=False)
    model.eval()
    x = torch.rand(1, 3, 448, 448)
    with torch.no_grad():
        cls, regs, patches = model(x)
    assert cls.shape == (1, 1, 1024) 
    assert regs.shape == (1, 1, 1024)
    assert patches.shape[0] == 1
    assert patches.shape[-1] == 1024