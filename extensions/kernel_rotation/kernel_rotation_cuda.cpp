/*
 * @Author: Lubo Wang
 * @Last Modified by:   Lubo Wang
 * @Last Modified time: 2022-08-09
 * @Email:  3018216177@tju.edu.cn
 */
#include<torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>


torch::Tensor features_rotate_forward(torch::Tensor features, torch::Tensor point_base, torch::Tensor weights, cudaStream_t stream);
torch::Tensor features_rotate_backward(torch::Tensor features, torch::Tensor point_base, torch::Tensor weights, torch::Tensor old_grad,  cudaStream_t stream);


torch::Tensor rotate_forward(torch::Tensor features, torch::Tensor point_base, torch::Tensor weights){
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    return features_rotate_forward(features, point_base, weights, stream);
}
torch::Tensor rotate_backward(torch::Tensor features, torch::Tensor old_grad, torch::Tensor point_base, torch::Tensor weights){
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    return features_rotate_backward(features, point_base, weights, old_grad, stream);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &rotate_forward,
        "get features rotated (CUDA)");
  m.def("backward", &rotate_backward,
        "features rotated backward (CUDA)");
}
    