/*
 * @Author: Lubo Wang
 * @Last Modified by:   Lubo Wang
 * @Last Modified time: 2022-08-09
 * @Email:  3018216177@tju.edu.cn
 */
#include <ATen/ATen.h>
#include<torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;


#define CUDA_NUM_THREADS 512

// Computer the number of threads needed in GPU
inline int get_n_threads(int n) {
  const int pow_2 = std::log(static_cast<double>(n)) / std::log(2.0);
  return max(min(1 << pow_2, CUDA_NUM_THREADS), 1);
}


//计算 8顶点 index
__device__ int * vtx_indexs(double centerf_d, double centerf_h, double centerf_w, int len_d, int len_h, int len_w){
    int * indexs = new int[8];

    int center_d = std::floor(centerf_d);
    int center_h = std::floor(centerf_h);
    int center_w = std::floor(centerf_w);

    indexs[0] = center_d * (len_h * len_w) + center_h * len_w + center_w;   //000
    indexs[1] = center_d * (len_h * len_w) + center_h * len_w + (center_w + 1);   //001
    indexs[2] = center_d * (len_h * len_w) + (center_h + 1) * len_w + center_w;   //010
    indexs[3] = center_d * (len_h * len_w) + (center_h + 1) * len_w + (center_w + 1);   //011
    indexs[4] = (center_d + 1) * (len_h * len_w) + center_h * len_w + center_w;   //100
    indexs[5] = (center_d + 1) * (len_h * len_w) + center_h * len_w + (center_w + 1);   //101
    indexs[6] = (center_d + 1) * (len_h * len_w) + (center_h + 1) * len_w + center_w;   //110
    indexs[7] = (center_d + 1) * (len_h * len_w) + (center_h + 1) * len_w + (center_w + 1);   //111

    if (center_d < 0 || center_d > len_d - 1){
        indexs[0] = -1;
        indexs[1] = -1;
        indexs[2] = -1;
        indexs[3] = -1;
    }
    if (center_d + 1 < 0 || center_d + 1 > len_d - 1){
        indexs[4] = -1;
        indexs[5] = -1;
        indexs[6] = -1;
        indexs[7] = -1;
    }
    if (center_h < 0 || center_h > len_h - 1){
        indexs[0] = -1;
        indexs[1] = -1;
        indexs[4] = -1;
        indexs[5] = -1;
    }
    if (center_h + 1 < 0 || center_h + 1 > len_h - 1){
        indexs[2] = -1;
        indexs[3] = -1;
        indexs[6] = -1;
        indexs[7] = -1;
    }
    if (center_w < 0 || center_w > len_w - 1){
        indexs[0] = -1;
        indexs[2] = -1;
        indexs[4] = -1;
        indexs[6] = -1;
    }
    if (center_w + 1 < 0 || center_w + 1 > len_w - 1){
        indexs[1] = -1;
        indexs[3] = -1;
        indexs[5] = -1;
        indexs[7] = -1;
    }

    return indexs;
}

//给定坐标 计算 index
__device__ int compute_index(int d, int h, int w, int len_h, int len_w) {
    return d * (len_h * len_w) + h * len_w + w;
}

//给定索引，判断邻域是否存在
__device__ double get_feature(const double * features, int index, int len){
    if(index >= 0 && index < len){
        return features[index];
    }
    else{
        return 0;
    }
}


__global__ void GetRotatedFeaturesForwardKernel(
    int len_d,
    int len_h,
    int len_w,
    const double *__restrict__ features,       //
    const double *__restrict__ weights,        // 27 * 8
    const double *__restrict__ points_base,    // 27 * 3
    double *__restrict__ features_rotated      // 需要reshape
    ){
    //batch_index
    // printf("features: %f",features);
    // blocks(batch_size, channel)
    int batch_index = blockIdx.x;
    int channel_index = blockIdx.y; 

    int index       = threadIdx.x;
    int stride      = blockDim.x;

    features += (batch_index * gridDim.y + channel_index) * len_d * len_h * len_w;
    features_rotated += (batch_index * gridDim.y + channel_index) * len_d * 3 * len_h * 3 * len_w * 3;

    for (int j = index; j < len_d * len_h * len_w; j += stride){
        // j 即为旋转前feature的index   对应feature_rotated  j * 27 
        // 找邻域 索引小于0或者大于15*9*15-1即为邻域不存在，设为0
        // 获得邻域周围特征

        //center坐标
        int f_d = j / (len_h * len_w);
        int f_h = j % (len_h * len_w) / len_w;
        int f_w = j % len_w;
        //邻域坐标
        double * neighbors = new double[27 * 3];
        for (int k = 0; k < 27; k ++){
            neighbors[k * 3 + 0] = f_d + points_base[k * 3 + 0];
            neighbors[k * 3 + 1] = f_h + points_base[k * 3 + 1];
            neighbors[k * 3 + 2] = f_w + points_base[k * 3 + 2];
        }
        
        int len_features = len_d * len_h * len_w;

        for (int m = 0; m < 27; m ++){
            int * indexs = vtx_indexs(neighbors[m * 3 + 0], neighbors[m * 3 + 1], neighbors[m * 3 + 2], len_d, len_h, len_w);
            //权重 插值
            // features_rotated[j * 27 + m] = features[indexs[0]] * weights[m * 8 + 0]
            //                              + features[indexs[1]] * weights[m * 8 + 1]
            //                              + features[indexs[2]] * weights[m * 8 + 2]
            //                              + features[indexs[3]] * weights[m * 8 + 3]
            //                              + features[indexs[4]] * weights[m * 8 + 4]
            //                              + features[indexs[5]] * weights[m * 8 + 5]
            //                              + features[indexs[6]] * weights[m * 8 + 6]
            //                              + features[indexs[7]] * weights[m * 8 + 7];
            features_rotated[j * 27 + m] = get_feature(features, indexs[0], len_features) * weights[m * 8 + 0]
                                         + get_feature(features, indexs[1], len_features) * weights[m * 8 + 1]
                                         + get_feature(features, indexs[2], len_features) * weights[m * 8 + 2]
                                         + get_feature(features, indexs[3], len_features) * weights[m * 8 + 3]
                                         + get_feature(features, indexs[4], len_features) * weights[m * 8 + 4]
                                         + get_feature(features, indexs[5], len_features) * weights[m * 8 + 5]
                                         + get_feature(features, indexs[6], len_features) * weights[m * 8 + 6]
                                         + get_feature(features, indexs[7], len_features) * weights[m * 8 + 7];
            delete indexs;
        }
        delete neighbors;
    }
}



__global__ void GetRotatedFeaturesBackwardKernel(
    int len_d,
    int len_h,
    int len_w,
    double *__restrict__ f_grad,
    const double *__restrict__ fr_grad,
    const double *__restrict__ points_base,
    const double *__restrict__ weights
    ){
    // blocks(batch_size, channel)
    int batch_index = blockIdx.x;
    int channel_index = blockIdx.y; 

    int index       = threadIdx.x;
    int stride      = blockDim.x;

    fr_grad += (batch_index * gridDim.y + channel_index) * len_d * 3 * len_h * 3 * len_w * 3;

    for (int j = index; j < len_d * len_h * len_w; j += stride){
        // j 即为旋转前feature的index   对应feature_rotated  j * 27 
        // 找邻域 索引小于0或者大于15*9*15-1即为邻域不存在，设为0
        // 获得邻域周围特征

        //center坐标
        int f_d = j / (len_h * len_w);
        int f_h = j % (len_h * len_w) / len_w;
        int f_w = j % len_w;
        //邻域坐标

        double * neighbors = new double[27 * 3];
        for (int k = 0; k < 27; k ++){
            neighbors[k * 3 + 0] = f_d + points_base[k * 3 + 0];
            neighbors[k * 3 + 1] = f_h + points_base[k * 3 + 1];
            neighbors[k * 3 + 2] = f_w + points_base[k * 3 + 2];
        }
        
        int len_features = len_d * len_h * len_w;

        for (int m = 0; m < 27; m ++){
            // 旋转前8顶点索引
            int * indexs = vtx_indexs(neighbors[m * 3 + 0], neighbors[m * 3 + 1], neighbors[m * 3 + 2], len_d, len_h, len_w);

            // atomicAdd(&f_grad[indexs[0]], fr_grad[j * 27 + m] * weights[m * 8 + 0]);
            // atomicAdd(&f_grad[indexs[1]], fr_grad[j * 27 + m] * weights[m * 8 + 1]);
            // atomicAdd(&f_grad[indexs[2]], fr_grad[j * 27 + m] * weights[m * 8 + 2]);
            // atomicAdd(&f_grad[indexs[3]], fr_grad[j * 27 + m] * weights[m * 8 + 3]);
            // atomicAdd(&f_grad[indexs[4]], fr_grad[j * 27 + m] * weights[m * 8 + 4]);
            // atomicAdd(&f_grad[indexs[5]], fr_grad[j * 27 + m] * weights[m * 8 + 5]);
            // atomicAdd(&f_grad[indexs[6]], fr_grad[j * 27 + m] * weights[m * 8 + 6]);
            // atomicAdd(&f_grad[indexs[7]], fr_grad[j * 27 + m] * weights[m * 8 + 7]);
            for (int n = 0; n < 8; n ++){
                if (indexs[n] >= 0 && indexs[n] < len_features){
                    atomicAdd(&f_grad[indexs[n]], fr_grad[j * 27 + m] * weights[m * 8 + n]);
                }
            }

            delete indexs;
        }
        delete neighbors;
    }
}


torch::Tensor features_rotate_forward(torch::Tensor features, torch::Tensor point_base, torch::Tensor weights, cudaStream_t stream){
    int batch_size = features.size(0);
    int len_d = features.size(2);
    int len_h = features.size(3);
    int len_w = features.size(4);
    

    int len_features = features.size(2) * features.size(3) * features.size(4);

    torch::Tensor features_rotated = torch::zeros({features.size(0), features.size(1), features.size(2) * features.size(3) * features.size(4) * 27}, torch::CUDA(torch::kFloat64));


    //TODO 计算权重  27 * 8
    

    // cout<<"type1:"<<features.type()<<endl;
    GetRotatedFeaturesForwardKernel<<<batch_size, get_n_threads(len_features), 0, stream>>>(
        len_d,
        len_h,
        len_w,
        features.data_ptr<double>(),
        weights.data_ptr<double>(),
        point_base.data_ptr<double>(),
        features_rotated.data_ptr<double>());

    // cout<<"type2:"<<features_rotated.type()<<endl;
    return features_rotated;
} 


torch::Tensor features_rotate_backward(torch::Tensor features, torch::Tensor point_base, torch::Tensor weights, torch::Tensor old_grad,  cudaStream_t stream){
    int batch_size = features.size(0);
    int len_d = features.size(2);
    int len_h = features.size(3);
    int len_w = features.size(4);
    

    int len_features = features.size(2) * features.size(3) * features.size(4);
    
    torch::Tensor grad = torch::zeros({features.size(0), features.size(1), features.size(2), features.size(3), features.size(4)}, torch::CUDA(torch::kFloat64));
    
    GetRotatedFeaturesBackwardKernel<<<batch_size, get_n_threads(len_features), 0, stream>>>(
        len_d,
        len_h,
        len_w,
        grad.data_ptr<double>(),
        old_grad.data_ptr<double>(),
        point_base.data_ptr<double>(),
        weights.data_ptr<double>());

    // cout<<"type3:"<<grad.type()<<endl;
    return grad;
} 