import torch
from .SHCNet import SHCNet
import paths
import sys


def model_generator(method, iter_stage=4):
    if method == 'SHCNet':
        model = SHCNet(iter_stage=iter_stage).cuda()
    else:
        print(f'Method {method} is not defined !!!!')
        sys.exit(0)
    if paths.pretrained_model_path is not None:
        print(f'load model from {paths.pretrained_model_path}')
        model = torch.load(paths.pretrained_model_path)

    return model
