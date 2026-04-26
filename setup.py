import os
import sys
from setuptools import setup, find_packages


def _make_extensions():
    """
    Bangun dua Cython/C extension modules.

    Jika Cython atau NumPy tidak tersedia saat build, fungsi ini
    mengembalikan list kosong dan paket terinstall dalam mode pure-Python
    (NumPy fallback tetap berfungsi penuh).

    Extension yang dihasilkan:
      entrap._numba_core   <- _numba_core.pyx + _c_src/cov_entropy.c
      entrap._intrinsic_dim <- _intrinsic_dim.pyx + _c_src/twonn.c
    """
    try:
        import numpy as np
        from Cython.Build import cythonize
        from setuptools import Extension
    except ImportError:
        return []

    here      = os.path.dirname(os.path.abspath(__file__))
    c_src_dir = os.path.join(here, "entrap", "_c_src")

    # Compiler flags
    # -O3           : tingkat optimisasi maksimal
    # -fno-finite-math-only : jaga HUGE_VAL/infinity agar tidak di-optimize away
    # /O2           : setara -O2 pada MSVC (tidak ada -fno-finite-math-only)
    if sys.platform == "win32":
        extra_compile_args = ["/O2"]
    else:
        extra_compile_args = ["-O3", "-fno-finite-math-only"]

    extensions = [
        Extension(
            name="entrap._numba_core",
            sources=[
                os.path.join(here, "entrap", "_numba_core.pyx"),
                os.path.join(c_src_dir, "cov_entropy.c"),
            ],
            include_dirs=[
                np.get_include(),
                c_src_dir,
            ],
            extra_compile_args=extra_compile_args,
            language="c",
        ),
        Extension(
            name="entrap._intrinsic_dim",
            sources=[
                os.path.join(here, "entrap", "_intrinsic_dim.pyx"),
                os.path.join(c_src_dir, "twonn.c"),
            ],
            include_dirs=[
                np.get_include(),
                c_src_dir,
            ],
            extra_compile_args=extra_compile_args,
            language="c",
        ),
    ]

    return cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck":    False,
            "wraparound":     False,
            "cdivision":      True,
            "nonecheck":      False,
            "initializedcheck": False,
        },
        annotate=False,
    )


setup(
    name="entrap",
    version="1.0.0",
    author="Muhammad Akmal Husain",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "hdbscan>=0.8.27",
        "ripser>=0.6.0",
        "kneed>=0.8.0",
        "joblib>=1.1.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        # Diperlukan saat build dari source untuk mengkompilasi C extensions.
        # Tidak dibutuhkan jika menginstall wheel yang sudah pre-compiled.
        "build": [
            "cython>=0.29.0",
        ],
    },
    package_data={
        "entrap": [
            "_c_src/*.h",
            "_c_src/*.c",
            "_numba_core.pyx",
            "_intrinsic_dim.pyx",
        ],
    },
    ext_modules=_make_extensions(),
)
