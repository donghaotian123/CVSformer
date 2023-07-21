import torch
import torch.nn as nn
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from extensions.kernel_rotation import KernelRotateFunction

class Rotation(nn.Module):
    def __init__(self):
        super(Rotation, self).__init__()

    def first_neighbor(self, x_y_z_offset, kernel_size):
        x = int(x_y_z_offset[0][0][0].item())
        y = int(x_y_z_offset[0][0][1].item())
        z = int(x_y_z_offset[0][0][2].item())
        neighbor=torch.zeros(kernel_size*kernel_size*kernel_size,3)
        neighbor_num=0

        for neighbor_x in range(x-1, x+2):
            for neighbor_y in range(y-1, y+2):
                for neighbor_z in range(z-1, z+2):
                
                    neighbor[neighbor_num]=torch.tensor([neighbor_x, neighbor_y, neighbor_z])
                    neighbor_num=neighbor_num+1
                       
        return neighbor

    
    
    def forward(self,featuremap):
        length = np.arange(featuremap.shape[2])
        width = np.arange(featuremap.shape[3])
        height = np.arange(featuremap.shape[4])
        a,b,c = np.meshgrid(length, width, height, indexing="ij")
        x_offset = torch.FloatTensor(a).view(-1, 1)
        y_offset = torch.FloatTensor(b).view(-1, 1)
        z_offset = torch.FloatTensor(c).view(-1, 1)
        x_y_z_offset = torch.cat((x_offset, y_offset, z_offset), 1).view(-1, 3).unsqueeze(0).repeat(featuremap.shape[0],1,1)
        kernel_size=3
        firstNeighbor = self.first_neighbor(x_y_z_offset, kernel_size)

        
        # z-axis
        rotation_matrix_45 = torch.tensor([[torch.cos(torch.tensor(np.pi/4)),torch.sin(torch.tensor(np.pi/4)),0],[-torch.sin(torch.tensor(np.pi/4)), torch.cos(torch.tensor(np.pi/4)),0],[0,0,1]])
        rotation_matrix_90 = torch.tensor([[torch.cos(torch.tensor(np.pi/2)),torch.sin(torch.tensor(np.pi/2)),0],[-torch.sin(torch.tensor(np.pi/2)), torch.cos(torch.tensor(np.pi/2)),0],[0,0,1]])
        rotation_matrix_135 = torch.tensor([[torch.cos(torch.tensor(np.pi*3/4)),torch.sin(torch.tensor(np.pi*3/4)),0],[-torch.sin(torch.tensor(np.pi*3/4)), torch.cos(torch.tensor(np.pi*3/4)),0],[0,0,1]])
       
        neighbor_rotation = torch.zeros(27,3)
        neighbor_unsq = firstNeighbor.unsqueeze(0)
        neighbor_sq_45 = torch.matmul(neighbor_unsq, rotation_matrix_45)
        neighbor_rotation_45 = neighbor_sq_45.squeeze(0)

        neighbor_sq_90 = torch.matmul(neighbor_unsq, rotation_matrix_90)
        neighbor_rotation_90 = neighbor_sq_90.squeeze(0)

        neighbor_sq_135 = torch.matmul(neighbor_unsq, rotation_matrix_135)
        neighbor_rotation_135 = neighbor_sq_135.squeeze(0)
       

        return neighbor_rotation_45, neighbor_rotation_90, neighbor_rotation_135, firstNeighbor

def compute_weights(points):
    # params: points 27*3
    len_points, len_xyz = points.shape
    weights = torch.zeros([len_points, 8], dtype=torch.float32)
    for i in range(len_points):
        x, y, z = points[i]
        fx = torch.floor(x)
        fy = torch.floor(y)
        fz = torch.floor(z)

        xd = x - fx
        yd = y - fy
        zd = z - fz

        weights[i][0] = (1-xd)*(1-yd)*(1-zd)
        weights[i][1] = (1-xd)*(1-yd)*zd
        weights[i][2] = (1-xd)*yd*(1-zd)
        weights[i][3] = (1-xd)*yd*zd
        weights[i][4] = xd*(1-yd)*(1-zd) 
        weights[i][5] = xd*(1-yd)*zd
        weights[i][6] = xd*yd*(1-zd)
        weights[i][7] = xd*yd*zd
    return weights

class KernelRotate(nn.Module):
    def __init__(self):
        super(KernelRotate, self).__init__()
    def forward(self, features, points_base, weights):
        return KernelRotateFunction.apply(features, points_base, weights)