import os
from distutils.core import setup

from setuptools import find_packages

install_requires = ["numpy", "scipy", "scikit-learn"]


here = os.path.abspath(os.path.dirname(__file__))
# Get __version__ variable
__version__ = "0.0.1"

setup(
    name="batchopt",
    version=__version__,  # NOQA
    description="batch optimization",
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
)
