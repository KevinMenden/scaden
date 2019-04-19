#!/usr/bin/env python

from setuptools import setup, find_packages
import sys

version = '0.1'

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
    packages = find_packages()

)