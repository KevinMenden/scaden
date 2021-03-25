# Scaden Blog
Apart from the changelog, this is a more informal section where I will inform about new features
that have been (or will be) implemented in Scaden.

# Scaden v1.1.0 - Performance Improvements and `scaden merge` tool (21.03.2021)

Scaden v1.1.0 brings significantly improved memory consumption for the data simulation step, which was a frequently asked for feature.
Now, instead of using about 4 GB of memory to simulate a small dataset, Scaden only uses 1 GB. Memory usage does not increase
with the number of datasets anymore. This will allow to create datasets from large collections of scRNA-seq datasets without 
needing excessive memory. Furthermore, Scaden now stores the simulated data in `.h5ad` format with the full list of genes.
This way you can simulate from a scRNA-seq dataset once and combine it with other datasets in the future. To help with this,
I've added the `scaden merge` command, which takes a list of datasets or a directory with `.h5ad` datasets and creates
a new training dataset from it.

