# Scaden Changelog

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

+ fixed Dockerfile (switched to pip installation)
+ added better error messages to `simulate` command
+ cleaned up dependencies

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