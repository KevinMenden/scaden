![Scaden](img/scaden_logo.png)
 
## Single-cell Assisted Deconvolutional Network

Scaden is a tool for bulk RNA-seq cell type deconvolutional that uses a deep neural network ensemble trained
on artificial bulk data simulated with scRNA-seq datasets. This method was developed in the [Genome Biology of Neurodegenerativ Diseases group](https://www.dzne.de/en/research/research-areas/fundamental-research/research-groups/heutink/research-areasfocus/)
at the [DZNE Tübingen](https://www.dzne.de/en/about-us/sites/tuebingen/) and the [Medical Systems Biology](https://www.uke.de/english/departments-institutes/institutes/medical-systems-biology/team/index.html) group at the [ZMNH](https://www.uke.de/english/departments-institutes/centers/center-for-molecular-neurobiology-hamburg-(zmnh)/index.html). The main author is [Kevin Menden](https://github.com/KevinMenden).

A pre-print describing the method is available on Biorxiv:
[Deep-learning-based cell composition analysis from tissue expression profiles](https://www.biorxiv.org/content/10.1101/659227v1)





## Changelog

#### Version 0.9.6
+ fixed Dockerfile (switched to pip installation)
+ added better error messages to `simulate` command

#### Version 0.9.5
+ added `scaden simulate` command to perform bulk simulation and training file creation
+ added `--seed` parameter to allow reproducible Scaden runs

#### Version 0.9.4
+ fixed dependencies (added python>=3.6 requirement)

#### Version 0.9.3
+ upgrade to Tensorflow 2
+ cleaned up dependencies

#### Version 0.9.2
+ RAM usage improvement

#### Version 0.9.1
+ Added automatic removal of duplicate genes in Mixture file 
+ Changed name of final prediction file
+ Added Scaden logo to main script


#### Version 0.9.0
This is the initial release version of Scaden. While this version contains full functionality for pre-processing, training and prediction, it does not
contain thorough error messages, plotting functionality and a solid helper function for generation training data. These are all features
planned for the release of v.1.0.0.
The core functionality of Scaden is, however, implemented and fully operational. Please check the [Usage](usage) section to learn how to use Scaden.