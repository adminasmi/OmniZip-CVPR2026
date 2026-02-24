from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mlp_relu2',
    ext_modules=[
        CUDAExtension('mlp_relu2', [
            'fused_mlp.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)}
)
