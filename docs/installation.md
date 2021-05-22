# Installation

Scaden be easily installed on a Linux system, and should also work on Mac. 
There are currently two options for installing Scaden, either using [Bioconda](https://bioconda.github.io/) or via [pip](https://pypi.org/).

## pip

To install Scaden via pip, simply run the following command:

`pip install scaden`

## Docker

If you don't want to install Scaden at all, but rather use a Docker container, we provide that as well.
For every release, we provide two version - one for CPU and one for GPU usage.
To pull the CPU container, use this command:

`docker pull ghcr.io/kevinmenden/scaden/scaden`

For the GPU container:

`docker pull ghcr.io/kevinmenden/scaden/scaden-gpu`

## Webtool (beta)

We now also provide a webtool for you:

[https://scaden.ims.bio](https://scaden.ims.bio)

It comes with pre-generated training datasets for several tissues. You only need to upload the epxression data for prediction. Please not that this is still in beta.
