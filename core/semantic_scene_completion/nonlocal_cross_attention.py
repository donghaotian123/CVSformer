import torch 
import torch.nn as nn
import torch.nn.functional as F
import os 
import math


class NonCrossAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 dimension=2,
                 sub_sample=True,
                 bn_layer=True):
        super(NonCrossAttention, self).__init__()
        assert dimension in [1, 2, 3]

        self.business_layer = []
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels


        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(2, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)
            
        self.business_layer.append(self.theta)
        self.business_layer.append(self.phi)
        self.business_layer.append(self.g)
        self.business_layer.append(self.W)

    def forward(self, x, y):
        batch_size = x.size(0) 

        '''
            Q
        '''
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1) #[B, N, C1]
        
        '''
            K
        '''
        phi_x = self.phi(y).view(batch_size, self.inter_channels, -1)

        '''
            V
        '''
        g_x = self.g(y).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1) 
      
        f = torch.matmul(theta_x, phi_x) 

        f_div_C = F.softmax(f, dim=-1) 

        y = torch.matmul(f_div_C, g_x) 
        y = y.permute(0, 2, 1).contiguous() 
        size = [batch_size, self.inter_channels] + list(x.size()[2:])
        y = y.view(size)  
        W_y = self.W(y)  
        z = W_y + x 
      
        return z


def main():
    GPU = '4' 
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x = torch.rand([16, 256, 15, 9, 15]).to(device)
    y = torch.rand([16, 256, 15, 9, 15]).to(device)
    attention = NonCrossAttention(256,inter_channels = 128,dimension = 3, sub_sample=False, gammas=[0,30,60,90]).to(device)
    output = attention(x, y)


if __name__ == "__main__":
    main()