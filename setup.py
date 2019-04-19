#!/usr/bin/env python

from setuptools import setup, find_packages

version = '1.0.0'

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = 'cdn',
    description = "Cell type deconvolution using single cell data",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    keywords = ['bioinformatics', 'deep learning', 'machine learning', 'single cell sequencing', 'deconvolution'],
    author = 'Kevin Menden',
    author_email = 'kevin.menden@t-online.de',
    scripts = ['cdn'],
    packages = find_packages(),
    install_requires = [
        'pandas',
        'numpy',
        'scikit-learn',
        'scipy',
        'tensorflow=1.10.0',
        'seaborn',
        'scanpy=1.2.2',
        'tqdm',
        'click'
    ]
)