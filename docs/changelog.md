# Scaden Changelog

## Version 1.1.2

* Changed datatype for data simulation back to `float32`

## Version 1.1.1

* Fixed bugs in scaden model definition [[#88](https://github.com/KevinMenden/scaden/issues/88)]
* removed installation instructions for bioconda as not functional at the moment [[#86](https://github.com/KevinMenden/scaden/issues/86)]
* Fixed bug in `scaden example` [[#85](https://github.com/KevinMenden/scaden/issues/85)]

## Version 1.1.0

* Reduced memory usage of `scaden simulate` significantly by performing simulation for one dataset at a time.
* Using `.h5ad` format to store simulated data
* Allow reading data in `.h5ad` format for improved performance (courtesy of @eboileau)
* Improved logging and using rich progress bar for training
* Gene subsetting is now done only when merging datasets, which will allow to generate different combinations
of simulated datasets
* Added `scaden merge` command which allows merging of previously created datasets  

### Version 1.0.2

* General improvement of logging using the 'rich' library for colorized output
* Added verification check for '--train_datasets' parameter to notify user of 
  unavailable datasets

### Version 1.0.1

* Made identification of datasets more robust to fix issue [#66](https://github.com/KevinMenden/scaden/issues/66)

### Version 1.0.0

* Rebuild Scaden model and training to use TF2 Keras API instead of the old compatibility functions 
* added `scaden example` command which allows to generate example data for test-running scaden and to inpstec the expected file format
* added more tests and checks input reading function in `scaden simulate`
* fixed bug in reading input data

### Version 0.9.6

* fixed Dockerfile (switched to pip installation)
* added better error messages to `simulate` command
* cleaned up dependencies

### v0.9.5

* added `--seed` parameter to allow reproducible Scaden runs
* added `scaden simulate` command to perform bulk simulation and training file creation
* changed CLI calling

### v0.9.4

* fixed dependencies (added python>=3.6 requirement)

### v0.9.3

* upgrade to tf2
* cleaned up dependencies

### v0.9.2

* small code refactoring
* RAM usage improvement

### v0.9.1

* added automatic removal of duplicate genes
* changed name of prediction file

### v0.9.0   

Initial release of the Scaden deconvolution package.

Commands:

* `scaden process`: Process a training dataset for training
* `scaden train`: Train a Scaden model
* `scaden predict`: Predict cell type compositions of a given sample