# encoding: utf-8

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

from config import config
from KernelRotateModule import Rotation
from KernelRotateModule import compute_weights
from KernelRotateModule import KernelRotate
from timm.models.layers import trunc_normal_
from caitModule import MultiViewBlock, OneViewBlock

import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from config import config

class SimpleRB(nn.Module):
    def __init__(self, in_channel, norm_layer, bn_momentum):
        super(SimpleRB, self).__init__()
        self.path = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            norm_layer(in_channel, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(in_channel, in_channel, kernel_size=3, padding=1, bias=False),
            norm_layer(in_channel, momentum=bn_momentum),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        conv_path = self.path(x)
        out = residual + conv_path
        out = self.relu(out)
        return out

class Bottleneck3D(nn.Module):

    def __init__(self, inplanes, planes, norm_layer, stride=1, dilation=[1, 1, 1], expansion=4, downsample=None,
                 fist_dilation=1, multi_grid=1,
                 bn_momentum=0.0003):
        super(Bottleneck3D, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 1, 3), stride=(1, 1, stride),
                               dilation=(1, 1, dilation[0]), padding=(0, 0, dilation[0]), bias=False)
        self.bn2 = norm_layer(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 1), stride=(1, stride, 1),
                               dilation=(1, dilation[1], 1), padding=(0, dilation[1], 0), bias=False)
        self.bn3 = norm_layer(planes, momentum=bn_momentum)
        self.conv4 = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1), stride=(stride, 1, 1),
                               dilation=(dilation[2], 1, 1), padding=(dilation[2], 0, 0), bias=False)
        self.bn4 = norm_layer(planes, momentum=bn_momentum)
        self.conv5 = nn.Conv3d(planes, planes * self.expansion, kernel_size=(1, 1, 1), bias=False)
        self.bn5 = norm_layer(planes * self.expansion, momentum=bn_momentum)

        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

        self.downsample2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, stride, 1), stride=(1, stride, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample3 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )
        self.downsample4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(stride, 1, 1), stride=(stride, 1, 1)),
            nn.Conv3d(planes, planes, kernel_size=1, stride=1, bias=False),
            norm_layer(planes, momentum=bn_momentum),
        )

    def forward(self, x):
        residual = x 

        out1 = self.relu(self.bn1(self.conv1(x))) 
        out2 = self.bn2(self.conv2(out1))
        out2_relu = self.relu(out2) 

        out3 = self.bn3(self.conv3(out2_relu))
        if self.stride != 1:
            out2 = self.downsample2(out2)
        out3 = out3 + out2
        out3_relu = self.relu(out3)

        out4 = self.bn4(self.conv4(out3_relu)) 
        if self.stride != 1:
            out2 = self.downsample3(out2)
            out3 = self.downsample4(out3)
        out4 = out4 + out2 + out3

        out4_relu = self.relu(out4)
        out5 = self.bn5(self.conv5(out4_relu))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out5 + residual
        out_relu = self.relu(out)

        return out_relu

class STAGE2(nn.Module):
    def __init__(self, class_num, norm_layer, resnet_out=2048, feature=512, ThreeDinit=True,
                 bn_momentum=0.1, pretrained_model=None, eval=False, freeze_bn=False):
        super(STAGE2, self).__init__()
        self.business_layer = []
        if eval:
            self.downsample = nn.Sequential(
                nn.Conv2d(resnet_out, feature, kernel_size=1, bias=False),
                nn.BatchNorm2d(feature, momentum=bn_momentum),
                nn.ReLU()
            )
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(resnet_out, feature, kernel_size=1, bias=False),
                norm_layer(feature, momentum=bn_momentum),
                nn.ReLU()
            )
        self.business_layer.append(self.downsample)
    
        self.semantic_layer1 = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=4, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature, momentum=bn_momentum),
            ), norm_layer=norm_layer),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]),
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.business_layer.append(self.semantic_layer1)
        self.semantic_layer2 = nn.Sequential(
            Bottleneck3D(feature, feature // 4, bn_momentum=bn_momentum, expansion=8, stride=2, downsample=
            nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(feature, feature * 2,
                          kernel_size=1, stride=1, bias=False),
                norm_layer(feature * 2, momentum=bn_momentum),
            ), norm_layer=norm_layer), # 128->256
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[1, 1, 1]), # 256
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[2, 2, 2]), # 256
            Bottleneck3D(feature * 2, feature // 2, bn_momentum=bn_momentum, norm_layer=norm_layer, dilation=[3, 3, 3]),
        )
        self.business_layer.append(self.semantic_layer2)
        self.classify_semantic = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=3, stride=2, padding=1, dilation=1,
                                   output_padding=1),
                norm_layer(feature, momentum=bn_momentum),
                nn.ReLU(inplace=False),
            ),
            nn.Sequential(
                nn.ConvTranspose3d(feature, feature, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
                norm_layer(feature, momentum=bn_momentum),
                nn.ReLU(inplace=False),
            ),
            nn.Sequential(
                nn.Dropout3d(.1),
                nn.Conv3d(feature, class_num, kernel_size=1, bias=True)
            ),
            nn.Sequential(
                nn.Dropout3d(.1),
                nn.Conv3d(feature, 2, kernel_size=1, bias=True)
            )]
        )
        self.business_layer.append(self.classify_semantic)
        self.oper_tsdf = nn.Sequential(
            nn.Conv3d(1, 3, kernel_size=3, padding=1, bias=False),
            norm_layer(3, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(3, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(64, feature, kernel_size=3, padding=1, bias=False),
            norm_layer(feature, momentum=bn_momentum),
            nn.ReLU(inplace=False),
        )
        self.oper_raw = nn.Sequential(
            nn.Conv3d(14, 3, kernel_size=3, padding=1, bias=False),
            norm_layer(3, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(3, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv3d(64, feature, kernel_size=3, padding=1, bias=False),
            norm_layer(feature, momentum=bn_momentum),
            nn.ReLU(inplace=False),
        )
        self.business_layer.append(self.oper_raw)
        self.business_layer.append(self.oper_tsdf)

        '''multi-directional-cross-attention'''
       
        self.num_branches = 4
        self.RotationBlock = RotationBlock(dim=256, num_branches = self.num_branches)
        self.business_layer += self.RotationBlock.business_layer

        self.k = 75 #cls token size 5*3*5

        self.MultiView = MultiView(iter=1, dim=256, num_branches=self.num_branches, k=self.k)
        self.OneViewBlock = OneViewBlock(dim=256, num_branches=self.num_branches, k=self.k)
        
        self.business_layer += self.MultiView.business_layer
        self.business_layer += self.OneViewBlock.business_layer

        self.cls_token = nn.ParameterList([nn.Parameter(torch.zeros(1, self.k, 256)) for i in range(self.num_branches)])
        self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, self.k+15*9*15, 256)) for i in range(self.num_branches)])
        self.business_layer.append(self.cls_token)
        self.business_layer.append(self.pos_embed)
        self.pos_drop = nn.Dropout(p=0.)
        for i in range(self.num_branches):
            if self.pos_embed[i].requires_grad:
                trunc_normal_(self.pos_embed[i], std=.02)
            trunc_normal_(self.cls_token[i], std=.02)
        
        if self.num_branches==4:
            self.conv_channel_proj = nn.Conv3d(in_channels=256*4,out_channels=256,kernel_size=3,padding=1)
        elif self.num_branches==3:
            self.conv_channel_proj = nn.Conv3d(in_channels=256*3,out_channels=256,kernel_size=3,padding=1)
        self.business_layer.append(self.conv_channel_proj)

    def forward(self, raw, tsdf, mapping):
        '''project 2D seg_result to 3D seg_result'''
        b, c, h, w = raw.shape
        print('bs:',b)
        raw = raw.view(b, c, h*w).permute(0, 2, 1)
        zerosVec = torch.zeros(b, 1, c).cuda()
        segVec =  torch.cat((raw, zerosVec), 1)
        segres = [torch.index_select(segVec[i], 0, mapping[i]) for i in range(b)]
        segres = torch.stack(segres).permute(0, 2, 1).contiguous().view(b, c, 60, 36, 60)
        raw = segres

        edge_raw = self.oper_raw(raw)         
        edge_tsdf = self.oper_tsdf(tsdf)     
        seg_fea = edge_tsdf + edge_raw
        semantic1 = self.semantic_layer1(seg_fea) + F.interpolate(seg_fea, size=[30, 18, 30])
        semantic2 = self.semantic_layer2(semantic1) 
        
        semantic2_down = semantic2
        B,C,D,H,W = semantic2.shape
        '''rotation'''
        outs = self.RotationBlock(semantic2_down) 

        '''multi-view-cross-attention'''

        feature_list = [] 
        outs_ = [outs[i].flatten(2).transpose(1,2)  for i in range(self.num_branches)] 
        for i in range(self.num_branches):
            tmp = outs_[i] 
            cls_tokens = self.cls_token[i].expand(b, -1, -1)  
            tmp = torch.cat((cls_tokens, tmp), dim=1)
            tmp = tmp + self.pos_embed[i] 
            tmp = self.pos_drop(tmp)
            feature_list.append(tmp)      
        
        semantic2 = self.OneViewBlock(outs, feature_list, key = 'vt_conv', branch = 4)

        up_sem1 = self.classify_semantic[0](semantic2) 
        up_sem1 = up_sem1 + semantic1
        up_sem2 = self.classify_semantic[1](up_sem1) 
        up_sem2 = up_sem2 + F.interpolate(up_sem1, size=[60, 36, 60], mode="trilinear", align_corners=True)

        up_sem2 = up_sem2 +edge_raw 

        pred_semantic = self.classify_semantic[2](up_sem2)
        return pred_semantic, outs

class MultiView(nn.Module):
    def __init__(self, iter=3, dim=256, num_branches=3, k=75):
        super(MultiView,self).__init__()
        self.business_layer = []
        self.blocks = nn.ModuleList()
        for _ in range(iter):
            blk = MultiViewBlock(dim, num_branches, k)
            self.business_layer += blk.business_layer
            self.blocks.append(blk)
        
    def forward(self, feature_list):
        for blk in self.blocks: # transfomer layers
            feature_list = blk(feature_list)
        return feature_list

class RotationBlock(nn.Module):
    def __init__(self, dim=256,num_branches=4):
        super(RotationBlock,self).__init__()
        self.business_layer = []
        self.rotation = Rotation()
        self.num_branches= num_branches
        self.rotationModule = KernelRotate()
        self.rotationConv1 = nn.Conv3d(dim, dim, kernel_size=3, padding=0, stride=3)
        self.rotationConv2 = nn.Conv3d(dim, dim, kernel_size=3, padding=0, stride=3)
        self.rotationConv3 = nn.Conv3d(dim, dim, kernel_size=3, padding=0, stride=3)
        self.business_layer.append(self.rotationConv1)
        self.business_layer.append(self.rotationConv2)
        self.business_layer.append(self.rotationConv3)

    def forward(self, semantic2_down):
        b_s2, c_s2, d_s2, h_s2, w_s2 = semantic2_down.shape
        points_base_45, points_base_90, points_base_135, point_base0 = self.rotation(semantic2_down)
        weights_45 = compute_weights(points_base_45)
        weights_90 = compute_weights(points_base_90)
        weights_135 = compute_weights(points_base_135)

        semantic2_r45 = self.rotationModule(semantic2_down, points_base_45.cuda(), weights_45.cuda()).reshape(b_s2, c_s2, 3, 3, -1)
        semantic2_r90 = self.rotationModule(semantic2_down, points_base_90.cuda(), weights_90.cuda()).reshape(b_s2, c_s2, 3, 3, -1)
        semantic2_r135 = self.rotationModule(semantic2_down, points_base_135.cuda(), weights_135.cuda()).reshape(b_s2, c_s2, 3, 3, -1)
        semantic2_r45 = self.rotationConv1(semantic2_r45).reshape(b_s2, c_s2, d_s2, h_s2, w_s2) # (16,256,7,4,7)
        semantic2_r90 = self.rotationConv2(semantic2_r90).reshape(b_s2, c_s2, d_s2, h_s2, w_s2)
        semantic2_r135 = self.rotationConv3(semantic2_r135).reshape(b_s2, c_s2, d_s2, h_s2, w_s2)
        if self.num_branches==4:
            return [semantic2_down, semantic2_r45,semantic2_r90,semantic2_r135] # 4 view
        elif self.num_branches==3:
            return [semantic2_r45,semantic2_r90,semantic2_r135]

class Network(nn.Module):
    def __init__(self, class_num, norm_layer, resnet_out=2048, feature=512, ThreeDinit=True,
                 bn_momentum=0.1, pretrained_model=None, eval=False, freeze_bn=False):
        super(Network, self).__init__()
        self.business_layer = []
        self.dilate = 2
        self.stage2 = STAGE2(class_num, norm_layer, resnet_out=resnet_out, feature=feature, ThreeDinit=ThreeDinit,
                             bn_momentum=bn_momentum, pretrained_model=pretrained_model, eval=eval, freeze_bn=freeze_bn)
        self.business_layer += self.stage2.business_layer

    def forward(self, ttt, tsdf, mmm):

        pp, ppp = self.stage2(ttt, tsdf, mmm)

        if self.training:
            return pp, ppp
        return pp, ppp

    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)


if __name__ == '__main__':
    pass


