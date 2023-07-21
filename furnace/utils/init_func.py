
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from caitModule import *
from network import *

def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (norm_layer, nn.LayerNorm)):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)

def group_weight_V1(model, lr):
    group_decay = []
    group_no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if name.endswith(".bias") or check_keywords_in_name(name, '.bn'):
            group_no_decay.append(param)
        else:
            group_decay.append(param)
    return [{'params': group_decay, 'lr':lr},
            {'params': group_no_decay, 'lr':lr,'weight_decay': 0.}]


def check_keywords_in_name(name, keywords='.bn'):
    isin = False
    if keywords in name:
        isin = True
    return isin
