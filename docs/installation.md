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
If you don't want to install Scaden at all, but rather use a Docker container, we provide that as well.
You can pull the [Scaden docker container](https://hub.docker.com/r/kevinmenden/scaden) with the following command (from Dockerhub):

`docker pull kevinmenden/scaden`

