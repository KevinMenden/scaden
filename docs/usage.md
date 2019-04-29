# Usage

For a typical deconvolution with Scaden you will have to perform three steps:

* pre-processing of training data
* training of Scaden model
* prediction

This assumes that you already have a training dataset. If not, Scaden contains functionality to create a dataset from one or several scRNA-seq datasets.
Please refer to the [data generation](##Training-data-generation) section for instructions on how to create training datasets.

Note that we already provide datasets for certain tissues. All available datasets are listed in the [Datasets](##Datasets) section. We will
update this section when new datasets are added. 

## Pre-processing
The first step is to pre-process your training data. For this you need your training data and the dataset you want to perform deconvolution on.
In this step, Scaden will create a new file for training which only contains the intersection of genes between the training and the prediction data.
Furthermore, the training data will be log2-transformed and scaled to the range [0,1]. Use the following command for pre-processing:
```
scaden process <training data> <prediction data>
```

## Training
Now that your data is set-up, you can start training a Scaden ensemble model. Scaden consists of three deep neural network models. By default,
each of them will be trained for 20,000 steps. You can train longer if you want, although we got good results with this number for datasets of 
around 30,000 samples. Use the following command to just train a model for 20,000 steps:

`scaden train <processed data>`

This will save the model parameters in your working directory. If you want to create a specific directory for your trained models instead,
and train for 30,00 steps, you can use this command:

`scaden train <processed data> --model_dir <model dir> --steps 30000`

You can also adjust the batch size and the learning rate, although we recommend using the default values. If you want to adjust them anyway, use these flages:

`--batch_size <batch size>`

`--learning_rate <learning rate>`

## Prediction 
Finally, after your model is trained, you can start the prediction. If you haven't specified any model directory and just trained a model
in your current directory, you can use the following command to perform the deconvolution: 

`scaden predict <prediction file>`

Scaden will then generate a file called 'cdn_predictions.txt' (this name will change in future releases) in your current directory. If the models were saved elsewhere,
you have to tell Scaden where to look for them:

`scaden predict <prediction file> --model_dir <model dir>`

You can also change the path and name of the output predictions file using the `outname` flag:

`--outname <path/to/output.txt`


## Training data generation

... coming soon ...