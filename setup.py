#!/usr/bin/env python3

from setuptools import setup, find_packages

version = "1.0.1"

with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

with open("LICENSE", encoding="UTF-8") as f:
    license = f.read()

setup(
    name="scaden",
    version=version,
    description="Cell type deconvolution using single cell data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=[
        "bioinformatics",
        "deep learning",
        "machine learning",
        "single cell sequencing",
        "deconvolution",
    ],
    author="Kevin Menden",
    author_email="kevin.menden@t-online.de",
    url="https://github.com/KevinMenden/scaden",
    license="MIT License",
    entry_points={"console_scripts": ["scaden=scaden.__main__:main"]},
    packages=find_packages(),
    include_package_data=True,
    python_requires=">3.6.0",
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "tensorflow>=2.0",
        "anndata",
        "tqdm",
        "click",
        "h5py~=2.10.0",
    ],
)
