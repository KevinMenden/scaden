"""
Main file for using CDN
"""

import argparse
from model.cdn import CDN
from model.functions import *
import tensorflow as tf


if __name__=="__main__":

    """
    Argument parsing section
    """
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="Which mode. One of [ train | predict | train_predict | process ]", default="train")
    parser.add_argument("--data", type=str, help="Path to dataset")
    parser.add_argument("--model", type=str, help="Model directory", default="/home/kevin/jobs")
    parser.add_argument("--name", type=str, help="Model name", default="cdn_model")
    parser.add_argument("--batch_size", type=int, help="Batch size. Default: 128", default=128)
    parser.add_argument("--learning_rate", "-l", type=float, help="Learning rate. Default: 0.0001", default=0.0001)
    parser.add_argument("--steps", type=int, help="Number of steps to use for training", default=1000)
    parser.add_argument("--processed_path", type=str, help="Out path for processed data", default="./processed.h5ad")
    parser.add_argument("--scaling", type=str, help="Scaling option. Default: log_min_max", default="log_min_max")
    parser.add_argument("--out", type=str, help="Path to store output in. Default: cwd", default="./")
    parser.add_argument("--training_data", type=str, help="Path to training dataset.", default=None)
    parser.add_argument("--outname", type=str, help="Name of the prediction output file. Default: cdn_predictions.txt", default="cdn_predictions.txt")
    # Parse args
    args = parser.parse_args()
    mode = args.mode
    data_path = args.data
    model_dir = args.model
    model_name = args.name
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_steps = args.steps
    processed_path = args.processed_path
    scaling_option = args.scaling
    out_dir = args.out
    training_data = args.training_data
    out_name = args.outname


    """
    TRAINING ONLY
    In training only mode, a CDN Model is trained only on the training data
    For later predictions, it might be the case that the set of training features does not
    entirely overlap
    """

    # Training mode
    if mode == "train":
        training(data_path=data_path,
                 model_dir=model_dir,
                 model_name=model_name,
                 batch_size=batch_size,
                 learning_rate=learning_rate,
                 num_steps=num_steps)

    """
    PREDICTION
    Use a trained CDN model to perform prediction on a given dataset
    This model can be trained either using training only mode or train and predict mode
    """
    # Prediction mode
    if mode == "predict":

        with tf.Session() as sess:
            cdn = CDN(sess=sess,
                      model_dir=model_dir,
                      model_name=model_name,
                      batch_size=batch_size,
                      learning_rate=learning_rate,
                      scaling=scaling_option)

            # Predict ratios
            cdn.predict(input_path=data_path, out_dir=out_dir, out_name=out_name, training_data=training_data)



    """
    TRAIN_PREDICT
    Use a prediction dataset to select the common set of genes to use as features, then train a CDN model 
    on these features
    After successfull training, perform prediction using this model
    """
    # Input-file specific training and predictions
    if mode == "train_predict":
        # Get signature genes overlap
        import scanpy.api as sc

        raw_input = sc.read_h5ad(training_data)
        sig_genes_complete = list(raw_input.var_names)
        sig_genes = get_signature_genes(input_path=data_path, sig_genes_complete=sig_genes_complete)

        # Pre-process data with new signature genes
        preprocess_h5ad_data(raw_input_path=training_data,
                             scaling_option=scaling_option,
                             processed_path=processed_path,
                             group_small=True,
                             signature_genes=True,
                             alt_sig_genes=sig_genes)

        # Perform training with new data
        with tf.Session() as sess:
            cdn = CDN(sess=sess,
                      model_dir=model_dir,
                      model_name=model_name,
                      batch_size=batch_size,
                      learning_rate=learning_rate,
                      num_steps=num_steps)
            # Start training
            cdn.train(input_path=processed_path)

        # Perform prediction
        tf.reset_default_graph()
        with tf.Session() as sess:
            cdn = CDN(sess=sess,
                      model_dir=model_dir,
                      model_name=model_name,
                      batch_size=batch_size,
                      learning_rate=learning_rate,
                      scaling=scaling_option)
            cdn.sig_genes=sig_genes

            # Predict ratios
            cdn.predict(input_path=data_path, out_dir=out_dir, out_name=out_name, training_data=training_data)


    # Pre-processing mode
    """
    PREPROCESS
    Use built-in functionality to preprocess a dataset (h5ad) format into
    correct scale
    """
    if mode == "process":
        preprocess_h5ad_data(raw_input_path=data_path,
                             scaling_option=scaling_option,
                             processed_path=processed_path,
                             group_small=True,
                             signature_genes=False)

