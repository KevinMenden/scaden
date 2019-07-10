# Usage

For a typical deconvolution with Scaden you will have to perform three steps:

* pre-processing of training data
* training of Scaden model
* prediction

This assumes that you already have a training dataset. If not, Scaden contains functionality to create a dataset from one or several scRNA-seq datasets.
Please refer to the [data generation](#training-data-generation) section for instructions on how to create training datasets.

Note that we already provide datasets for certain tissues. All available datasets are listed in the [Datasets](datasets) section. We will
update this section when new datasets are added. 

## Pre-processing
The first step is to pre-process your training data. For this you need your training data and the dataset you want to perform deconvolution on.
In this step, Scaden will create a new file for training which only contains the intersection of genes between the training and the prediction data.
Furthermore, the training data will be log2-transformed and scaled to the range [0,1]. Use the following command for pre-processing:

```console
scaden process <training data> <prediction data>
```

## Training
Now that your data is set-up, you can start training a Scaden ensemble model. Scaden consists of three deep neural network models. By default,
each of them will be trained for 20,000 steps. You can train longer if you want, although we got good results with this number for datasets of 
around 30,000 samples. Use the following command to just train a model for 20,000 steps:


```console
scaden train <processed data>
```

This will save the model parameters in your working directory. If you want to create a specific directory for your trained models instead,
and train for 30,00 steps, you can use this command:


```console
scaden train <processed data> --model_dir <model dir> --steps 30000
```


You can also adjust the batch size and the learning rate, although we recommend using the default values. If you want to adjust them anyway, use these flages:


```console
--batch_size <batch size>

--learning_rate <learning rate>
```

## Prediction 
Finally, after your model is trained, you can start the prediction. If you haven't specified any model directory and just trained a model
in your current directory, you can use the following command to perform the deconvolution: 

```console
scaden predict <prediction file>
```

Scaden will then generate a file called 'cdn_predictions.txt' (this name will change in future releases) in your current directory. If the models were saved elsewhere,
you have to tell Scaden where to look for them:

```console
scaden predict <prediction file> --model_dir <model dir>
```


You can also change the path and name of the output predictions file using the `outname` flag:

```console
--outname <path/to/output.txt
```

## File Formats
For Scaden to work properly, your input files have to be correctly formatted. As long as you use Scadens inbuilt functionality to generate the training data, you should have no problem 
with formatting there. The prediction file, however, you have to format yourself. This should be a file of shape m X n, where m are your features (genes) and n your samples. So each row corresponds to 
a gene, and each column to a sample. Leave the column name for the genes empy (just put a `\t` there). This is a rather standard format to store gene expression tables, so you should have not much work assuring that the
format fits.

Your data can either be raw counts or normalized, just make sure that they are not in logarithmic space already. When loading a prediction file, Scaden applies its scaling procedure to it, which involves taking the logarithm of your counts.
So as long as they are not already in logarithmic space, Scaden will be able to handle both raw and normalized counts / expression values.

## Training data generation
As version 0.9.0 is a pre-release version of Scaden, generation of artificial bulk RNA-seq data is not nicely implemented yet, but Scaden still ships with all the scripts to do it. 
There are generally three steps you have to do to generate training data, given you have a suitable scRNA-seq dataset:

* Generate normalized counts and associated cell type labels
* Generate artificial bulk samples
* Merge all samples into a h5ad file

I'll quickly explain how to go about that currently. I plan to make this workflow much easier in the future.

#### scRNA-seq data processing
The first step is to process your scRNA-seq dataset(s) you want to use for training. I used Scanpy for this, and would therefore
recommend to do the same, but you can of course use other software for this purpose. I've uploaded the scripts I used to preprocess
the data used for the Scaden paper [here](https://doi.org/10.6084/m9.figshare.8234030.v1). Mainly you have to normalize your count data
and create a file containing the cell type labels. The file for the cell type labels should be of size (n x 2), where n is the number of cells 
you have in your data. The two columns correspond to a label for your cells, and a 'Celltype' column. It is important to name
