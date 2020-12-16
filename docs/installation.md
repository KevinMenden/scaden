# Installation
Scaden be easily installed on a Linux system, and should also work on Mac. 
There are currently two options for installing Scaden, either using [Bioconda](https://bioconda.github.io/) or via [pip](https://pypi.org/).


## pip
To install Scaden via pip, simply run the following command:

`pip install scaden`


## Bioconda
You can also install Scaden via bioconda, using::

`conda install -c bioconda scaden`


## Docker
If you don't want to install Scaden at all, but rather use a Docker container, we provide that as well.
You can pull the [Scaden docker container](https://hub.docker.com/r/kevinmenden/scaden) with the following command (from Dockerhub):

`docker pull kevinmenden/scaden`

## Webtool (beta)
We now also provide a webtool for you:

[https://scaden.ims.bio](https://scaden.ims.bio)

It comes with pre-generated training datasets for several tissues. You only need to upload the epxression data for prediction. Please not that this is still in beta.
