from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="dmesh2",
    packages=['mindiffdt'],
    ext_modules=[
        CUDAExtension(
            name="mindiffdt._C",
            sources=[
                # "cuda_mindiffdt/render.cu",
                # "cuda_mindiffdt/rl.cu",
                # "cuda_mindiffdt/rdt.cu",
                "cuda_mindiffdt/manifold.cpp",
                "mindiffdt/mindiffdt.cu",
                "mindiffdt/ext.cpp"],
            extra_compile_args={"nvcc": [
                "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"),
                "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "cgal_wrapper/"),
                "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda_mindiffdt/"),
            ]},
            library_dirs=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "cgal_wrapper/"),],
            libraries=["cgal_wrapper", "gmp", "mpfr",],
        )
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)