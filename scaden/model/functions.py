"""
Functions used for the scaden model
"""
import collections
from anndata import read_h5ad
import numpy as np
import tensorflow as tf
from sklearn import preprocessing as pp
import pandas as pd




def dummy_labels(m, labels):
    """
    Create dummy labels needed for building the graph correctly
    :param m:
    :param labels:
    :return:
    """
    n_l = len(labels)
    return np.zeros((m, n_l), dtype="float32")

def sample_scaling(x, scaling_option):
    """
    Apply scaling of data
    :param x:
    :param scaling_option:
    :return:
    """

    if scaling_option == "log_min_max":
        # Bring in log space
        x = np.log2(x + 1)

        # Normalize data
        mms = pp.MinMaxScaler(feature_range=(0, 1), copy=True)

        # it scales features so transpose is needed
        x = mms.fit_transform(x.T).T

    return x



def preprocess_h5ad_data(raw_input_path, processed_path, scaling_option="log_min_max",  sig_genes=None):
    """
    Preprocess raw input data for the model
    :param raw_input_path:
    :param scaling_option:
    :param group_small:
    :param signature_genes:
    :return:
    """
    print("Pre-processing raw data ...")
    raw_input = read_h5ad(raw_input_path)

    print("Subsetting genes ...")
    # Select features go use
    raw_input = raw_input[:, sig_genes]

    print("Scaling using " + str(scaling_option))
    # Scaling
    raw_input.X = sample_scaling(raw_input.X, scaling_option)

    print("Writing to disk ...")
    # Write processed data to disk
    raw_input.write(processed_path)
    print("Data pre-processing done.")


def get_signature_genes(input_path, sig_genes_complete, var_cutoff=0.1):
    """
    Get overlap between signature genes and available genes
    :param input_path:
    :param sig_genes_complete:
    :return: new sig_genes
    """
    data = pd.read_table(input_path, index_col=0)
    keep = data.var(axis=1) > var_cutoff
    data = data.loc[keep]
    available_genes = list(data.index)
    new_sig_genes = list(set(available_genes).intersection(sig_genes_complete))
    n_sig_genes = len(new_sig_genes)
    print(f"Found {n_sig_genes} common genes.")
    return new_sig_genes




