from distutils.core import setup

from setuptools import find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='real_estate_image_type',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    install_requires=requirements,
    python_requires='>=3.7',
)