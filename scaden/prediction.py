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


def prediction(model_dir, data_path, out_name, seed=0):
    """
    Perform prediction using a trained scaden ensemble
    :param model_dir: the directory containing the models
    :param data_path: the path to the gene expression file
    :param out_name: name of the output prediction file
    :return:
    """

    # Small model predictions
    cdn256 = Scaden(model_dir=model_dir + "/m256",
                    model_name='m256',
                    seed=seed,
                    hidden_units=M256_HIDDEN_UNITS,
                    do_rates=M256_DO_RATES)
    # Predict ratios
    preds_256 = cdn256.predict(input_path=data_path,
                               out_name='scaden_predictions_m256.txt')

    # Mid model predictions
    cdn512 = Scaden(model_dir=model_dir + "/m512",
                    model_name='m512',
                    seed=seed,
                    hidden_units=M512_HIDDEN_UNITS,
                    do_rates=M512_DO_RATES)
    # Predict ratios
    preds_512 = cdn512.predict(input_path=data_path,
                               out_name='scaden_predictions_m512.txt')

    # Large model predictions
    cdn1024 = Scaden(model_dir=model_dir + "/m1024",
                     model_name='m1024',
                     seed=seed,
                     hidden_units=M1024_HIDDEN_UNITS,
                     do_rates=M256_DO_RATES)
    # Predict ratios
    preds_1024 = cdn1024.predict(input_path=data_path,
                                 out_name='scaden_predictions_m1024.txt')

    # Average predictions
    preds = (preds_256 + preds_512 + preds_1024) / 3
    preds.to_csv(out_name, sep="\t")
