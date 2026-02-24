import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="frequency_table",
        sources=["frequency_table.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-w"]
     ),
    Extension(
        name="arithmetic_coder",
        sources=["arithmetic_coder.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-w"]
     ),
    Extension(
        name="bitstreams",
        sources=["bitstreams.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-w"]
    )
]


setup(
    name="ArithmeticCoderProject",
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
)

