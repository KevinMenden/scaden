#!/usr/bin/env python

from setuptools import setup, find_packages

version = '0.9.1'

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name = 'scaden',
    version = version,
    description = "Cell type deconvolution using single cell data",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    keywords = ['bioinformatics', 'deep learning', 'machine learning', 'single cell sequencing', 'deconvolution'],
    author = 'Kevin Menden',
    author_email = 'kevin.menden@t-online.de',
    url='https://github.com/KevinMenden/scaden',
    license = license,
    scripts = ['scripts/scaden'],
    packages = find_packages(),
    include_package_data = True,
    install_requires = [
        'pandas==0.21',
        'numpy==1.14.5',
        'scikit-learn',
        'scipy',
        'seaborn',
        'tensorflow==1.12.1',
        'matplotlib',
        'scanpy==1.2.2',
        'tqdm',
        'click'
    ]
)