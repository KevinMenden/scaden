"""
scaden Main functionality

Contains code to
- process a training datasets
- train a model
- perform predictions

"""

# Imports
import tensorflow as tf
import scanpy.api as sc
from scaden.model.architectures import architectures
from scaden.model.scaden import Scaden
from scaden.model.functions import *

"""
PARAMETERS
"""
# ==========================================#

# Extract architectures
M256_HIDDEN_UNITS = architectures['m256'][0]
M512_HIDDEN_UNITS = architectures['m512'][0]
M1024_HIDDEN_UNITS = architectures['m1024'][0]
M256_DO_RATES = architectures['m256'][1]
M512_DO_RATES = architectures['m512'][1]
M1024_DO_RATES = architectures['m1024'][1]


# ==========================================#


def training(data_path, train_datasets, model_dir, batch_size, learning_rate, num_steps):
    """
    Perform training of three a scaden model ensemble consisting of three different models
    :param model_dir:
    :param batch_size:
    :param learning_rate:
    :param num_steps:
    :return:
    """
    # Convert training datasets
    if train_datasets == '':
        train_datasets = []
    else:
        train_datasets = train_datasets.split()
    print("Training on: " + str(train_datasets))


    # M256 model training
    print("Training M256 Model ...")
    tf.reset_default_graph()
    with tf.Session() as sess:
        cdn256 = Scaden(sess=sess,
                     model_dir=model_dir+"/m256",
                     model_name='m256',
                     batch_size=batch_size,
                     learning_rate=learning_rate,
                     num_steps=num_steps)
        cdn256.hidden_units = M256_HIDDEN_UNITS
        cdn256.do_rates = M256_DO_RATES
        cdn256.train(input_path=data_path, train_datasets=train_datasets)

    # Training of mid model
    print("Training M512 Model ...")
    tf.reset_default_graph()
    with tf.Session() as sess:
        cdn512 = Scaden(sess=sess,
                     model_dir=model_dir+"/m512",
                     model_name='m512',
                     batch_size=batch_size,
                     learning_rate=learning_rate,
                     num_steps=num_steps)
        cdn512.hidden_units = M512_HIDDEN_UNITS
        cdn512.do_rates = M512_DO_RATES
        cdn512.train(input_path=data_path, train_datasets=train_datasets)

    # Training of large model
    print("Training M1024 Model ...")
    tf.reset_default_graph()
    with tf.Session() as sess:
        cdn1024 = Scaden(sess=sess,
                      model_dir=model_dir+"/m1024",
                      model_name='m1024',
                      batch_size=batch_size,
                      learning_rate=learning_rate,
                      num_steps=num_steps)
        cdn1024.hidden_units = M1024_HIDDEN_UNITS
        cdn1024.do_rates = M1024_DO_RATES
        cdn1024.train(input_path=data_path, train_datasets=train_datasets)

    print("Training finished.")


def prediction(model_dir, data_path, out_name):
    """
    Perform prediction using a trained scaden ensemble
    :param model_dir: the directory containing the models
    :param data_path: the path to the gene expression file
    :param out_name: name of the output prediction file
    :return:
    """

    # Small model predictions
    tf.reset_default_graph()
    with tf.Session() as sess:
        cdn256 = Scaden(sess=sess,
                     model_dir=model_dir + "/m256",
                     model_name='m256')
        cdn256.hidden_units = M256_HIDDEN_UNITS
        cdn256.do_rates = M256_DO_RATES

        # Predict ratios
        preds_256 = cdn256.predict(input_path=data_path,  out_name='scaden_predictions_m256.txt')


    # Mid model predictions
    tf.reset_default_graph()
    with tf.Session() as sess:
        cdn512 = Scaden(sess=sess,
                     model_dir=model_dir+"/m512",
                     model_name='m512')
        cdn512.hidden_units = M512_HIDDEN_UNITS
        cdn512.do_rates = M512_DO_RATES

        # Predict ratios
        preds_512 = cdn512.predict(input_path=data_path, out_name='scaden_predictions_m512.txt')

    # Large model predictions
    tf.reset_default_graph()
    with tf.Session() as sess:
        cdn1024 = Scaden(sess=sess,
                      model_dir=model_dir+"/m1024",
                      model_name='m1024')
        cdn1024.hidden_units = M1024_HIDDEN_UNITS
        cdn1024.do_rates = M1024_DO_RATES

        # Predict ratios
        preds_1024 = cdn1024.predict(input_path=data_path, out_name='scaden_predictions_m1024.txt')

    # Average predictions
    preds = (preds_256 + preds_512 + preds_1024) / 3
    preds.to_csv(out_name, sep="\t")


def processing(data_path, training_data, processed_path):
    """
    Process a training dataset to contain only the genes also available in the prediction data
    :param data_path: path to prediction data
    :param training_data: path to training data (h5ad file)
    :param processed_path: name of processed file
    :return:
    """
    # Get the common genes (signature genes)
    raw_input = sc.read_h5ad(training_data)
    sig_genes_complete = list(raw_input.var_names)
    sig_genes = get_signature_genes(input_path=data_path, sig_genes_complete=sig_genes_complete)

    # Pre-process data with new signature genes
    preprocess_h5ad_data(raw_input_path=training_data,
                         processed_path=processed_path,
                         sig_genes=sig_genes)
