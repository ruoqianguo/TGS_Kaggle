from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("rlen", ["rlen.pyx", "rlen_core.c"])],
    include_dirs=[numpy.get_include()]
)