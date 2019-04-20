"""
cdn
author: Kevin Menden, DZNE Tübingen

This is the main file for executing the cdn program.
"""

# imports
import argparse
from cdn.model.cdn import CDN
from cdn.model.functions import *
import tensorflow as tf
from cdn import cdn_main

if __name__=="__main__":

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="Which mode. One of [ train | predict | process ]", default="train")
    parser.add_argument("--data", type=str, help="Path to dataset")
    parser.add_argument("--model", type=str, help="Model directory", default="/home/kevin/jobs")
    parser.add_argument("--name", type=str, help="Model name", default="cdn_model")
    parser.add_argument("--batch_size", type=int, help="Batch size. Default: 128", default=128)
    parser.add_argument("--learning_rate", "-l", type=float, help="Learning rate. Default: 0.0001", default=0.0001)
    parser.add_argument("--steps", type=int, help="Number of steps to use for training", default=1000)
    parser.add_argument("--processed_path", type=str, help="Out path for processed data", default="./processed.h5ad")
    parser.add_argument("--training_data", type=str, help="The training dataset that should be processed.")
    parser.add_argument("--scaling", type=str, help="Scaling option. Default: log_min_max", default="log_min_max")
    parser.add_argument("--outname", type=str, help="Name of the prediction output file. Default: cdn_predictions.txt", default="cdn_predictions.txt")
    parser.add_argument("--datasets", type=str, nargs='+', help="Datasets to use for training", default=['data6k', 'data8k', 'donorA', 'donorC', 'GSE65133'])

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
    out_name = args.outname
    train_datasets = args.datasets
    training_data = args.training_data

    # Training mode
    if mode == "train":
        cdn_main.training(data_path=data_path,
                          train_datasets=train_datasets,
                          model_dir=model_dir,
                          batch_size=batch_size,
                          learning_rate=learning_rate,
                          num_steps=num_steps)

    # Prediction mode
    if mode == "predict":
        cdn_main.prediction(model_dir=model_dir,
                            data_path=data_path,
                            out_name=out_name)

    # Processing mode
    if mode == "process":
        cdn_main.processing(data_path=data_path,
                            training_data=training_data,
                            processed_path=processed_path)

