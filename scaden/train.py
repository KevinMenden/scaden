"""
scaden Main functionality

Contains code to
- process a training datasets
- train a model
- perform predictions

"""

# Imports
import tensorflow as tf
from anndata import read_h5ad
from scaden.model.architectures import architectures
from scaden.model.scaden import Scaden
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


def training(data_path,
             train_datasets,
             model_dir,
             batch_size,
             learning_rate,
             num_steps,
             seed=0):
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
        train_datasets = train_datasets.split(',')
        print(f"Training on: {train_datasets}")

    # Training of M256 model
    print("Training M256 Model ...")
    cdn256 = Scaden(model_dir=model_dir + "/m256",
                    model_name='m256',
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    num_steps=num_steps,
                    seed=seed,
                    hidden_units=M256_HIDDEN_UNITS,
                    do_rates=M512_DO_RATES)
    cdn256.train(input_path=data_path, train_datasets=train_datasets)
    del cdn256

    # Training of M512 model
    print("Training M512 Model ...")
    cdn512 = Scaden(model_dir=model_dir + "/m512",
                    model_name='m512',
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    num_steps=num_steps,
                    seed=seed,
                    hidden_units=M512_HIDDEN_UNITS,
                    do_rates=M512_DO_RATES)
    cdn512.train(input_path=data_path, train_datasets=train_datasets)
    del cdn512

    # Training of M1024 model
    print("Training M1024 Model ...")
    cdn1024 = Scaden(model_dir=model_dir + "/m1024",
                     model_name='m1024',
                     batch_size=batch_size,
                     learning_rate=learning_rate,
                     num_steps=num_steps,
                     seed=seed,
                     hidden_units=M1024_HIDDEN_UNITS,
                     do_rates=M1024_DO_RATES)
    cdn1024.train(input_path=data_path, train_datasets=train_datasets)
    del cdn1024

    print("Training finished.")