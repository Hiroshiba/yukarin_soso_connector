import platform
from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize
from Cython.Distutils import build_ext

ext_modules = [
    Extension(
        name="each_cpp_forwarder",
        sources=["each_cpp_forwarder.pyx"],
        language="c++",
        libraries=["each_cpp_forwarder"],
        library_dirs=["../build/Debug"] if platform.system() == "Windows" else [],
    )
]

setup(
    name="each_cpp_forwarder",
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(ext_modules),
    include_dirs=[
        numpy.get_include(),
    ],
)
