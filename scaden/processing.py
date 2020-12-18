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
from scaden.model.functions import get_signature_genes, preprocess_h5ad_data

"""
PARAMETERS
"""
# ==========================================#

def processing(data_path, training_data, processed_path, var_cutoff):
    """
    Process a training dataset to contain only the genes also available in the prediction data
    :param data_path: path to prediction data
    :param training_data: path to training data (h5ad file)
    :param processed_path: name of processed file
    :return:
    """
    # Get the common genes (signature genes)
    raw_input = read_h5ad(training_data)
    sig_genes_complete = list(raw_input.var_names)
    sig_genes = get_signature_genes(input_path=data_path, sig_genes_complete=sig_genes_complete, var_cutoff=var_cutoff)

    # Pre-process data with new signature genes
    preprocess_h5ad_data(raw_input_path=training_data,
                         processed_path=processed_path,
                         sig_genes=sig_genes)
