import pytest
import torch

import timm
from timm.models.cpubone_original import CPUBoneBackbone as CPUBoneBackboneOrig
from timm.models.cpubone import CPUBoneBackbone


def test_backbone_equal():
    torch.manual_seed(42)
    backbone = CPUBoneBackbone([16, 32, 64, 128, 256], [0, 1, 1, 3, 4])
    torch.manual_seed(42)
    backbone_orig = CPUBoneBackboneOrig([16, 32, 64, 128, 256], [0, 1, 1, 3, 4])

    torch.manual_seed(42)
    x = torch.randn(1, 3, 224, 224)
    torch.manual_seed(42)
    out = backbone(x)
    torch.manual_seed(42)
    out_orig = backbone_orig(x)

    print(type(backbone), type(backbone_orig))
    print(out.keys(), out_orig.keys())
    for key in out.keys():
        out_lay = out[key]
        out_lay_orig = out_orig[key]
        print(key, out_lay.shape, out_lay_orig.shape)
        # allclose instead of equal: F.scaled_dot_product_attention uses fused kernels whose
        # float accumulation order differs from the original manual implementation (~1e-5 max diff)
        assert torch.allclose(out_lay, out_lay_orig, atol=1e-4), f"Equality test failed in output '{key}'"


def test_create_model():
    model = timm.create_model("cpubone_b0")
    model.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 1000)
    assert not torch.isnan(out).any()


def test_create_model_num_classes():
    model = timm.create_model("cpubone_nano", num_classes=10)
    model.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 10)