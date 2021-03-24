# Scaden Blog
Apart from the changelog, this is a more informal section where I will inform about new features
that have been (or will be) implemented in Scaden.

# Scaden v1.1.0 - Performance Improvements (21.03.2021)

Scaden v1.1.0 brings significantly improved memory consumption for the data simulation step, which was a asked for 
quite frequently. Now, instead of using about 4 GB of memory to simulate a small dataset, Scaden only uses 1 GB. This will
allow to create datasets from large collections of scRNA-seq datasets without needing excessive memory. Furthermore,
Scaden now stores the simulated data in `.h5ad` format with the full list of genes. This way you can simulate from a
scRNA-seq dataset once and combine it with other datasets in the future.
