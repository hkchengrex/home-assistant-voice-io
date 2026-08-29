from __future__ import annotations

import sys

from setuptools import Extension, setup


compile_args = ["/O2"] if sys.platform == "win32" else ["-O3"]

setup(
    ext_modules=[
        Extension(
            "ha_voice._dtw_native",
            sources=["src/ha_voice/_dtw_native.c"],
            extra_compile_args=compile_args,
            optional=True,
        )
    ]
)
