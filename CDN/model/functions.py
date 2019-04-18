"""
Functions used for the CDN model
"""
import collections
import scanpy.api as sc
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



def preprocess_h5ad_data(raw_input_path, scaling_option, processed_path,
                         group_small=True, signature_genes=True, alt_sig_genes=None):
    """
    Preprocess raw input data for the model
    :param raw_input_path:
    :param scaling_option:
    :param group_small:
    :param signature_genes:
    :return:
    """
    print("Pre-processing raw data ...")
    raw_input = sc.read_h5ad(raw_input_path)

    print("Grouping small cell types ...")
    # Group unknown celltypes together if desired
    if group_small:
        for cell_type in raw_input.uns['unknown'][1:]:
            raw_input.obs[raw_input.uns['unknown'][0]] += raw_input.obs[cell_type]
            del raw_input.obs[cell_type]

        new_cell_types = list(raw_input.obs.columns)
        new_cell_types.remove('ds')
        new_cell_types.remove('batch')
        raw_input.uns['cell_types'] = np.array(new_cell_types)

    print("Subsetting genes ...")
    # Select features go use
    if signature_genes:
        if alt_sig_genes == None:
            raw_input = raw_input[:, raw_input.uns['sig_genes']]
        else:
            raw_input = raw_input[:, alt_sig_genes]

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
    print(data.shape)
    available_genes = list(data.index)
    new_sig_genes = list(set(available_genes).intersection(sig_genes_complete))
    return new_sig_genes




