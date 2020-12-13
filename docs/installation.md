# Installation
Scaden be easily installed on a Linux system, and should also work on Mac. 
There are currently two options for installing Scaden, either using [Bioconda](https://bioconda.github.io/) or via [pip](https://pypi.org/).


## Bioconda
Installation via Bioconda is the preferred route of installation, and we highly recommend using conda. To install Scaden, use:

`conda install -c bioconda scaden`

It is always recommended to create a separate conda environment for installation.


## pip
If you don't want to use conda, you can also install Scaden using pip:

`pip install scaden`


## Docker
A docker container with Scaden installed is also available from [biocontainers](https://biocontainers.pro/#/).
You can download containers for all Scaden version here:
[https://biocontainers.pro/#/tools/scaden](https://biocontainers.pro/#/tools/scaden)
If you will also find Singularity images there.

## Webtool (beta)
We now also provide a webtool for you:

[https://scaden.ims.bio](https://scaden.ims.bio)

It comes with pre-generated training datasets for several tissues. You only need to upload the epxression data for prediction. Please not that this is still in beta.
