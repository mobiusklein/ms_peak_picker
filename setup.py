import sys
import traceback
import os
import platform

from setuptools import setup, Extension as _Extension, find_packages

from distutils.command.build_ext import build_ext
from distutils.errors import (CCompilerError, DistutilsExecError,
                              DistutilsPlatformError)


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


# with_openmp = has_option('with-openmp')
no_openmp = has_option('no-openmp')
debug_symbols = has_option("debug")

with_openmp = not no_openmp

include_diagnostics = has_option("include-diagnostics")
force_cythonize = has_option("force-cythonize")

use_python_implementation = has_option("pure-python")

print("Building with OpenMP? %s" % with_openmp)


def configure_openmp(ext):
    # http://www.microsoft.com/en-us/download/confirmation.aspx?id=2092 was required.
    if os.name == 'nt' and with_openmp:
        ext.extra_compile_args.append("/openmp")
    elif platform.system() == 'Darwin':
        pass
    elif with_openmp:
        ext.extra_compile_args.append("-fopenmp")
        ext.extra_link_args.append("-fopenmp")


def Extension(*args, **kwargs):
    ext = _Extension(*args, **kwargs)
    if debug_symbols:
        if os.name == 'nt':
            ext.extra_compile_args.append("-Zi")
            ext.extra_compile_args.append("-Ox")
            ext.extra_link_args.append("-debug:full")
        else:
            ext.extra_compile_args.append("-g3")
            ext.extra_compile_args.append("-O0")
    return ext

def OpenMPExtension(*args, **kwargs):
    ext = Extension(*args, **kwargs)
    configure_openmp(ext)
    return ext


def make_cextensions():
    import numpy
    macros = []
    try:
        from Cython.Build import cythonize
        cython_directives = {
            'embedsignature': True,
            "profile": include_diagnostics
        }
        if include_diagnostics:
            macros.append(("CYTHON_TRACE_NOGIL", "1"))
        print("Using Directives", cython_directives)
        extensions = cythonize([
            OpenMPExtension(
                name="ms_peak_picker._c.peak_statistics", sources=["src/ms_peak_picker/_c/peak_statistics.pyx"],
                include_dirs=[numpy.get_include()], define_macros=macros),
            Extension(name='ms_peak_picker._c.peak_set', sources=["src/ms_peak_picker/_c/peak_set.pyx"]),
            Extension(name='ms_peak_picker._c.fft_patterson_charge_state',
                      sources=["src/ms_peak_picker/_c/fft_patterson_charge_state.pyx"],
                      include_dirs=[numpy.get_include()], define_macros=macros),
            Extension(name="ms_peak_picker._c.search", sources=["src/ms_peak_picker/_c/search.pyx"],
                      include_dirs=[numpy.get_include()], define_macros=macros),
            Extension(name="ms_peak_picker._c.peak_index", sources=["src/ms_peak_picker/_c/peak_index.pyx"],
                      include_dirs=[numpy.get_include()], define_macros=macros),
            Extension(name='ms_peak_picker._c.double_vector', sources=["src/ms_peak_picker/_c/double_vector.pyx"]),
            Extension(name='ms_peak_picker._c.size_t_vector', sources=[
                      "src/ms_peak_picker/_c/size_t_vector.pyx"]),
            Extension(name='ms_peak_picker._c.interval_t_vector', sources=[
                      "src/ms_peak_picker/_c/interval_t_vector.pyx"]),
            Extension(name='ms_peak_picker._c.smoother', sources=["src/ms_peak_picker/_c/smoother.pyx"],
                      include_dirs=[numpy.get_include()], define_macros=macros),
            OpenMPExtension(name='ms_peak_picker._c.scan_averaging',
                            sources=['src/ms_peak_picker/_c/scan_averaging.pyx'],
                            include_dirs=[numpy.get_include()], define_macros=macros),
            Extension(name='ms_peak_picker._c.fticr_denoising', sources=['src/ms_peak_picker/_c/fticr_denoising.pyx'],
                      include_dirs=[numpy.get_include()], define_macros=macros),
            Extension(name="ms_peak_picker._c.peak_picker", sources=['src/ms_peak_picker/_c/peak_picker.pyx'],
                      include_dirs=[numpy.get_include()], define_macros=macros),
        ], compiler_directives=cython_directives, force=force_cythonize)
    except ImportError:
        extensions = ([
            OpenMPExtension(
                name="ms_peak_picker._c.peak_statistics", sources=["src/ms_peak_picker/_c/peak_statistics.c"],
                include_dirs=[numpy.get_include()], define_macros=macros),
            Extension(name='ms_peak_picker._c.peak_set', sources=[
                      "src/ms_peak_picker/_c/peak_set.c"], define_macros=macros),
            Extension(name='ms_peak_picker._c.fft_patterson_charge_state',
                      sources=["src/ms_peak_picker/_c/fft_patterson_charge_state.c"],
                      include_dirs=[numpy.get_include()], define_macros=macros),
            Extension(name="ms_peak_picker._c.search", sources=["src/ms_peak_picker/_c/search.c"],
                      include_dirs=[numpy.get_include()], define_macros=macros),
            Extension(name="ms_peak_picker._c.peak_index", sources=["src/ms_peak_picker/_c/peak_index.c"],
                      include_dirs=[numpy.get_include()], define_macros=macros),
            Extension(name='ms_peak_picker._c.double_vector', sources=[
                      "src/ms_peak_picker/_c/double_vector.c"], define_macros=macros),
            Extension(name='ms_peak_picker._c.size_t_vector', sources=[
                "src/ms_peak_picker/_c/size_t_vector.c"]),
            Extension(name='ms_peak_picker._c.interval_t_vector', sources=[
                      "src/ms_peak_picker/_c/interval_t_vector.c"]),
            Extension(name='ms_peak_picker._c.smoother', sources=["src/ms_peak_picker/_c/smoother.c"],
                      include_dirs=[numpy.get_include()], define_macros=macros),
            OpenMPExtension(name='ms_peak_picker._c.scan_averaging',
                            sources=['src/ms_peak_picker/_c/scan_averaging.c'],
                            include_dirs=[numpy.get_include()], define_macros=macros),
            Extension(name='ms_peak_picker._c.fticr_denoising', sources=['src/ms_peak_picker/_c/fticr_denoising.c'],
                      include_dirs=[numpy.get_include()], define_macros=macros)
        ])
    return extensions


ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError)
if sys.platform == 'win32':
    # 2.6's distutils.msvc9compiler can raise an IOError when failing to
    # find the compiler
    ext_errors += (IOError,)


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


with open("src/ms_peak_picker/version.py") as version_file:
    version = None
    for line in version_file.readlines():
        if "version = " in line:
            version = line.split(" = ")[1].replace("\"", "").strip()
            print("Version is: %r" % (version,))
            break
    else:
        print("Cannot determine version")


install_requires = [
    "numpy>=2.0.0; python_version >= '3.9'",
    "numpy; python_version < '3.9'",
    "scipy",
    "six",
]


def run_setup(include_cext=True):
    setup(
        name="ms_peak_picker",
        description="A library to pick peaks from mass spectral data",
        long_description="A library to pick peaks from mass spectral data",
        version=version,
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        zip_safe=False,
        install_requires=install_requires,
        ext_modules=make_cextensions() if include_cext else None,
        cmdclass=cmdclass,
        author="Joshua Klein",
        author_email="jaklein@bu.edu",
        maintainer="Joshua Klein",
        maintainer_email="jaklein@bu.edu",
        include_package_data=True,
        classifiers=[
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Topic :: Education",
            "Topic :: Scientific/Engineering :: Bio-Informatics",
            "Topic :: Scientific/Engineering :: Chemistry",
            "Topic :: Software Development :: Libraries",
        ],
        license="License :: OSI Approved :: Apache Software License",
    )


try:
    run_setup(True)
except Exception as exc:
    import traceback
    traceback.print_exc()
    if use_python_implementation:
        run_setup(False)
        status_msgs(
            "WARNING: The C extension could not be compiled, " +
            "speedups are not enabled.",
            "Plain-Python build succeeded."
        )
    else:
        raise
