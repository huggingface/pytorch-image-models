import pytest
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_pil_image
import torch
from timm.data.patchmix import PatchMix


def cpu_and_cuda():
    import pytest  # noqa

    return ("cpu", pytest.param("cuda", marks=pytest.mark.needs_cuda))


def needs_cuda(test_func):
    import pytest  # noqa

    return pytest.mark.needs_cuda(test_func)


@needs_cuda
@pytest.mark.parametrize("batch_size ", (4, 7))
@pytest.mark.parametrize("prob", (1.0, 0.5, 0.0))
@pytest.mark.parametrize("mix_num", (1, 2, 3))
@pytest.mark.parametrize("device", cpu_and_cuda())
def test_patchmix(batch_size, prob, mix_num, device):
    data_set = datasets.ImageFolder(
        root='data/',
        transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
    )
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, num_workers=4, shuffle=True)

    patchmix = PatchMix(10, prob, mix_num, 16)

    for images, _ in data_loader:
        b, c, w, h = images.shape
        images = images.to(device)
        target = torch.arange(batch_size).to(device)
        org_img = images.permute(1, 0, 2, 3).reshape(c, b * w, h)
        mix_img, mo_target, mm_target = patchmix(images, target)
        mix_img = mix_img.permute(1, 0, 2, 3).reshape(c, b * w, h)
        result = torch.cat([org_img, mix_img], dim=-1)
        to_pil_image(result).save(f"bs{batch_size}_p{prob}_n{mix_num}_{device}.png")
        print(target)
        print(mo_target)
        print(mm_target)
        break
