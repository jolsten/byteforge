import numpy
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


class optional_build_ext(_build_ext):
    """Allow C extension build to fail gracefully."""

    def run(self):
        try:
            _build_ext.run(self)
        except Exception:
            self._warn_unavailable()

    def build_extension(self, ext):
        try:
            _build_ext.build_extension(self, ext)
        except Exception:
            self._warn_unavailable()

    def _warn_unavailable(self):
        print("*" * 70)
        print("WARNING: byteforge C extension could not be compiled.")
        print("Falling back to pure Python (slower).")
        print("*" * 70)


setup(
    cmdclass={"build_ext": optional_build_ext},
    ext_modules=[
        Extension(
            "byteforge._c.ufunc",
            sources=["src/byteforge/_c/ufunc.c"],
            include_dirs=[numpy.get_include(), "src/byteforge/_c"],
        ),
    ],
)
