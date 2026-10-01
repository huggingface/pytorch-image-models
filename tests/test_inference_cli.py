import json

import pytest
import torch
import torch.nn as nn

pytest.importorskip('pandas')
PIL_Image = pytest.importorskip('PIL.Image')


class _TinyClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained_cfg = {}
        self.fc = nn.Linear(3, num_classes)

    def forward(self, x):
        return self.fc(x.mean(dim=(2, 3)))


@pytest.mark.parametrize('results_format', ['json', 'json-split', 'parquet', 'csv'])
def test_inference_cli_custom_filename_col(monkeypatch, tmp_path, results_format):
    import inference

    if results_format == 'parquet':
        pytest.importorskip('pyarrow')
    img_dir = tmp_path / 'images' / 'c0'
    img_dir.mkdir(parents=True)
    for i in range(3):
        PIL_Image.new('RGB', (16, 16), (40 * i, 10, 10)).save(img_dir / f'im{i}.jpg')

    monkeypatch.setattr(inference, 'create_model', lambda *a, **kw: _TinyClassifier(kw['num_classes']))
    monkeypatch.setattr('sys.argv', [
        'inference.py', '--data-dir', str(tmp_path / 'images'), '--model', 'resnet18', '--num-classes', '5',
        '--img-size', '16', '--workers', '0', '--batch-size', '2', '--topk', '1', '--device', 'cpu',
        '--label-type', 'none', '--filename-col', 'image', '--results-format', results_format,
        '--results-dir', str(tmp_path / 'out'), '--results-file', 'preds', '--no-console-results',
    ])
    inference.main()

    ext = {'json': '.json', 'json-split': '.json', 'parquet': '.parquet', 'csv': '.csv'}[results_format]
    out_file = tmp_path / 'out' / f'preds{ext}'
    assert out_file.exists()
    if results_format == 'json':
        assert sorted(json.loads(out_file.read_text())) == ['im0.jpg', 'im1.jpg', 'im2.jpg']
