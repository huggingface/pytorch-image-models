import pytest
import torch

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
        assert torch.equal(out_lay, out_lay_orig), f"Equality test failed in output '{key}'"