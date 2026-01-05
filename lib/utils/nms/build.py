from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='_ext_nms',
    ext_modules=[
        CUDAExtension(
            name='_ext.nms',
            sources=[
                'src/nms_cuda.c',
                'src/nms_cuda_kernel.cu'
            ],
            extra_compile_args={'nvcc': ['-arch=sm_120']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)