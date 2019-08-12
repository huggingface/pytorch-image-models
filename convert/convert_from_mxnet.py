import argparse
import hashlib
import os

import mxnet as mx
import gluoncv
import torch
from timm import create_model

parser = argparse.ArgumentParser(description='Convert from MXNet')
parser.add_argument('--model', default='all', type=str, metavar='MODEL',
                    help='Name of model to train (default: "all"')


def convert(mxnet_name, torch_name):
    # download and load the pre-trained model
    net = gluoncv.model_zoo.get_model(mxnet_name, pretrained=True)

    # create corresponding torch model
    torch_net = create_model(torch_name)

    mxp = [(k, v) for k, v in net.collect_params().items() if 'running' not in k]
    torchp = list(torch_net.named_parameters())
    torch_params = {}

    # convert parameters
    # NOTE: we are relying on the fact that the order of parameters
    # are usually exactly the same between these models, thus no key name mapping
    # is necessary. Asserts will trip if this is not the case.
    for (tn, tv), (mn, mv) in zip(torchp, mxp):
        m_split = mn.split('_')
        t_split = tn.split('.')
        print(t_split, m_split)
        print(tv.shape, mv.shape)

        # ensure ordering of BN params match since their sizes are not specific
        if m_split[-1] == 'gamma':
            assert t_split[-1] == 'weight'
        if m_split[-1] == 'beta':
            assert t_split[-1] == 'bias'

        # ensure shapes match
        assert all(t == m for t, m in zip(tv.shape, mv.shape))

        torch_tensor = torch.from_numpy(mv.data().asnumpy())
        torch_params[tn] = torch_tensor

    # convert buffers (batch norm running stats)
    mxb = [(k, v) for k, v in net.collect_params().items() if any(x in k for x in ['running_mean', 'running_var'])]
    torchb = [(k, v) for k, v in torch_net.named_buffers() if 'num_batches' not in k]
    for (tn, tv), (mn, mv) in zip(torchb, mxb):
        print(tn, mn)
        print(tv.shape, mv.shape)

        # ensure ordering of BN params match since their sizes are not specific
        if 'running_var' in tn:
            assert 'running_var' in mn
        if 'running_mean' in tn:
            assert 'running_mean' in mn
            
        torch_tensor = torch.from_numpy(mv.data().asnumpy())
        torch_params[tn] = torch_tensor

    torch_net.load_state_dict(torch_params)
    torch_filename = './%s.pth' % torch_name
    torch.save(torch_net.state_dict(), torch_filename)
    with open(torch_filename, 'rb') as f:
        sha_hash = hashlib.sha256(f.read()).hexdigest()
    final_filename = os.path.splitext(torch_filename)[0] + '-' + sha_hash[:8] + '.pth'
    os.rename(torch_filename, final_filename)
    print("=> Saved converted model to '{}, SHA256: {}'".format(final_filename, sha_hash))


def map_mx_to_torch_model(mx_name):
    torch_name = mx_name.lower()
    if torch_name.startswith('se_'):
        torch_name = torch_name.replace('se_', 'se')
    elif torch_name.startswith('senet_'):
        torch_name = torch_name.replace('senet_', 'senet')
    elif torch_name.startswith('inceptionv3'):
        torch_name = torch_name.replace('inceptionv3', 'inception_v3')
    torch_name = 'gluon_' + torch_name
    return torch_name


ALL = ['resnet18_v1b', 'resnet34_v1b', 'resnet50_v1b', 'resnet101_v1b', 'resnet152_v1b',
       'resnet50_v1c', 'resnet101_v1c', 'resnet152_v1c', 'resnet50_v1d', 'resnet101_v1d', 'resnet152_v1d',
       #'resnet50_v1e', 'resnet101_v1e', 'resnet152_v1e',
       'resnet50_v1s', 'resnet101_v1s', 'resnet152_v1s', 'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_64x4d',
       'se_resnext50_32x4d', 'se_resnext101_32x4d', 'se_resnext101_64x4d', 'senet_154', 'inceptionv3']


def main():
    args = parser.parse_args()

    if not args.model or args.model == 'all':
        for mx_model in ALL:
            torch_model = map_mx_to_torch_model(mx_model)
            convert(mx_model, torch_model)
    else:
        mx_model = args.model
        torch_model = map_mx_to_torch_model(mx_model)
        convert(mx_model, torch_model)


if __name__ == '__main__':
    main()
