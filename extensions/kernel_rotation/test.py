# @Author: Lubo Wang
# @Last Modified by:   Lubo Wang
# @Last Modified time: 2022-08-09
# @Email:  3018216177@tju.edu.cn
import os
import sys
import torch
import torch.nn as nn
import unittest

from torch.autograd import gradcheck

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from extensions.kernel_rotation import KernelRotateFunction

class KernelRotate(nn.Module):
    def __init__(self):
        super(KernelRotate, self).__init__()
    def forward(self, features, points_base, weights):
        return KernelRotateFunction.apply(features, points_base, weights)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input = torch.rand(16, 256, 15,9,15)
    input.requires_grad = True
    point_base = torch.rand(27,3)
    # point_base.requires_grad = True
    point_base = point_base * 5 % 2
    weights = torch.rand(27, 3)
    model = KernelRotate().to(device)
    rotation = model(input.to(device), point_base.to(device), weights.to(device))
    print("1")
if __name__ == '__main__':
    GPU = '5' 
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    main()
