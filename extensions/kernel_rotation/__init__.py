# @Author: Lubo Wang
# @Last Modified by:   Lubo Wang
# @Last Modified time: 2022-08-09
# @Email:  3018216177@tju.edu.cn
import os 
import torch
import kernel_rotation

class KernelRotateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, points_base, weights):
        # print(features.dtype)
        features_output = kernel_rotation.forward(features, points_base, weights)
        
        # print("11111")
        # print(variables.dtype)
        ctx.save_for_backward(features, points_base, weights)
        return features_output

    @staticmethod
    def backward(ctx, grad_output):
        features, points_base, weights = ctx.saved_tensors
        grad = kernel_rotation.backward(features, grad_output, points_base, weights)
        return grad, None, None

# def main():
#     GPU = '3' 
#     os.environ["CUDA_VISIBLE_DEVICES"] = GPU
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     input = torch.rand([16,15,9,15,256], dtype=torch.float32, requires_grad=True).to(device)
#     point_base = torch.rand([3,3,3,3], dtype=torch.float32, requires_grad=True)
#     point_base = (point_base % 2).to(device)
#     print(torch.autograd.gradcheck(KernelRotateFunction.apply, (input, point_base,), eps=1e-3))

    
# if __name__ == "__main__":
#     main()