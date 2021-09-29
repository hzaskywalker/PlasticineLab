from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ChamferDistance_eric',
    ext_modules=[
        CUDAExtension('ChamferDistance_eric', [
            "/".join(__file__.split('/')[:-1] + ['chamfer_distance.cpp']),
            "/".join(__file__.split('/')[:-1] + ['chamfer_distance.cu']),
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })