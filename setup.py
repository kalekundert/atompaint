from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

setup(
        ext_modules=[
            Pybind11Extension(
                name='atompaint.datasets._voxelize',
                sources=[
                    'atompaint/datasets/_voxelize.cc',
                ],
                include_dirs=[
                        'atompaint/vendored/Eigen',
                        'atompaint/vendored/overlap',
                ],
                cxx_std=20,
            ),
        ],
)
