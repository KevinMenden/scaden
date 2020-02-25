#!/usr/bin/env python3

from setuptools import setup, find_packages

version = '0.9.3'

with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

with open('LICENSE', encoding="UTF-8") as f:
    license = f.read()

setup(
    name='scaden',
    version=version,
    description="Cell type deconvolution using single cell data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=['bioinformatics', 'deep learning', 'machine learning', 'single cell sequencing', 'deconvolution'],
    author='Kevin Menden',
    author_email='kevin.menden@t-online.de',
    url='https://github.com/KevinMenden/scaden',
    license="MIT License",
    scripts=['scripts/scaden'],
    packages=find_packages(),
    include_package_data=True,
    install_requires = [
        'pandas==0.21',
        'numpy==1.14.5',
        'scikit-learn',
        'scipy',
        'seaborn',
        'tensorflow>=2.0',
        'matplotlib',
        'scanpy==1.2.2',
        'tqdm',
        'click'
    ]
)
