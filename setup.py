import sys
import setuptools
from setuptools import setup, find_packages


__version__ = "0.0.0"


setup(
    name="lecun1989repro",
    version=__version__,
    description="Reproducing LeCun et al., 1989",
    packages=setuptools.find_packages(),
    install_requires=["torch", "torchvision", "numpy", "matplotlib", "tensorboardX"],
)
