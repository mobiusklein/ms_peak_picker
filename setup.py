import sys
import traceback
import os

from setuptools import setup, Extension, find_packages
import numpy

try:
    from Cython.Build import cythonize
    extensions = cythonize([
        Extension(name="ms_peak_picker._peak_statistics", sources=["ms_peak_picker/_peak_statistics.pyx"],
                  include_dirs=[numpy.get_include()]),
        Extension(name='ms_peak_picker._peak_set', sources=["ms_peak_picker/_peak_set.pyx"])
        ])
except ImportError:
    extensions = ([
        Extension(name="ms_peak_picker._peak_statistics", sources=["ms_peak_picker/_peak_statistics.c"],
                  include_dirs=[numpy.get_include()]),
        Extension(name='ms_peak_picker._peak_set', sources=["ms_peak_picker/_peak_set.c"])
        ])


from distutils.command.build_ext import build_ext
from distutils.errors import (CCompilerError, DistutilsExecError,
                              DistutilsPlatformError)

ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError)
if sys.platform == 'win32':
    # 2.6's distutils.msvc9compiler can raise an IOError when failing to
    # find the compiler
    ext_errors += (IOError,)

c_ext = "pyx"
try:
    from Cython.Build import cythonize
except:
    c_ext = "c"


def has_option(name):
    try:
        sys.argv.remove('--%s' % name)
        return True
    except ValueError:
        pass
    # allow passing all cmd line options also as environment variables
    env_val = os.getenv(name.upper().replace('-', '_'), 'false').lower()
    if env_val == "true":
        return True
    return False


class BuildFailed(Exception):

    def __init__(self):
        self.cause = sys.exc_info()[1]  # work around py 2/3 different syntax

    def __str__(self):
        return str(self.cause)


class ve_build_ext(build_ext):
    # This class allows C extension building to fail.

    def run(self):
        try:
            build_ext.run(self)
        except DistutilsPlatformError:
            traceback.print_exc()
            raise BuildFailed()

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except ext_errors:
            traceback.print_exc()
            raise BuildFailed()
        except ValueError:
            # this can happen on Windows 64 bit, see Python issue 7511
            traceback.print_exc()
            if "'path'" in str(sys.exc_info()[1]):  # works with both py 2/3
                raise BuildFailed()
            raise

cmdclass = {}

cmdclass['build_ext'] = ve_build_ext


def status_msgs(*msgs):
    print('*' * 75)
    for msg in msgs:
        print(msg)
    print('*' * 75)


def run_setup(include_cext=True):
    setup(
        name='ms_peak_picker',
        description='A library to pick peaks from mass spectral data',
        long_description='A library to pick peaks from mass spectral data',
        version="0.1.0",
        packages=find_packages(),
        zip_safe=False,
        install_requires=['numpy'],
        ext_modules=extensions if include_cext else None,
        cmdclass=cmdclass,
        maintainer='Joshua Klein',
        maintainer_email="jaklein@bu.edu",
        classifiers=[
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Topic :: Education',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Topic :: Scientific/Engineering :: Chemistry',
            'Topic :: Software Development :: Libraries'
        ],
        license='License :: OSI Approved :: Apache Software License')

try:
    run_setup(True)
except Exception as exc:
    run_setup(False)

    status_msgs(
        "WARNING: The C extension could not be compiled, " +
        "speedups are not enabled.",
        "Plain-Python build succeeded."
    )
