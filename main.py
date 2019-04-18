"""
Main file for using CDN
"""

import argparse
from CDN.model.cdn import CDN
from CDN.model.functions import *
import tensorflow as tf

if __name__=="__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="Which mode. One of [ train | predict | train_predict | process ]", default="train")
    parser.add_argument("--data", type=str, help="Path to dataset")
    parser.add_argument("--model", type=str, help="Model directory", default="/home/kevin/jobs")
    parser.add_argument("--batch_size", type=int, help="Batch size. Default: 128", default=128)
    parser.add_argument("--learning_rate", "-l", type=float, help="Learning rate. Default: 0.0001", default=0.0001)
    parser.add_argument("--steps", type=int, help="Number of steps to use for training", default=1000)
    parser.add_argument("--processed_path", type=str, help="Out path for processed data", default="./processed.h5ad")
    parser.add_argument("--out", type=str, help="Path to store output in. Default: cwd", default="./")
    parser.add_argument("--training_data", type=str, help="Path to training dataset.", default=None)
    parser.add_argument("--outname", type=str, help="Name of the prediction output file. Default: cdn_predictions.txt", default="cdn_predictions.txt")
    parser.add_argument("--datasets", type=str, nargs='+', help="Datasets to use for training", default=['data6k', 'data8k', 'donorA', 'donorC', 'GSE65133'])
    parser.add_argument("--units", type=int, nargs='+', help="Layer sizes for NN. (4 Numbers)", default=[256, 128, 64, 32])
    parser.add_argument("--dropout", type=float, nargs='+', help="Dropout percentages for each layer (4 numbers)", default=[0, 0, 0, 0])

    # Parse args
    args = parser.parse_args()
    mode = args.mode
    data_path = args.data
    model_dir = args.model
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_steps = args.steps
    processed_path = args.processed_path
    scaling_option = args.scaling
    out_dir = args.out
    training_data = args.training_data
    out_name = args.outname
    train_datasets = args.datasets
    units = args.units
    dropout = args.dropout

    # Training mode
    if mode == "train":
        with tf.Session() as sess:
            cdn = CDN(sess=sess,
                      model_dir=model_dir,
                      model_name=model_name,
                      batch_size=batch_size,
                      learning_rate=learning_rate,
                      num_steps=num_steps)

            # Start training
            cdn.train(input_path=data_path, train_datasets=train_datasets)

    # Prediction mode
    if mode == "predict":

        with tf.Session() as sess:
            cdn = CDN(sess=sess,
                      model_dir=model_dir,
                      model_name=model_name,
                      batch_size=batch_size,
                      learning_rate=learning_rate)

            # Predict ratios
            cdn.predict(input_path=data_path, out_dir=out_dir, out_name=out_name, training_data=training_data)


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
            cdn.train(input_path=processed_path, train_datasets=train_datasets)

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
    if mode == "process":
        preprocess_h5ad_data(raw_input_path=data_path,
                             scaling_option=scaling_option,
                             processed_path=processed_path,
                             group_small=True,
                             signature_genes=False)

