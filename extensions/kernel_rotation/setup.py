# @Author: Lubo Wang
# @Last Modified by:   Lubo Wang
# @Last Modified time: 2022-08-09
# @Email:  3018216177@tju.edu.cn
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='kernel_rotation',
    ext_modules=[
        CUDAExtension('kernel_rotation', [
            'kernel_rotation_cuda.cpp',
            'kernel_rotation_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
