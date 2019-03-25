"""
Functions used for the CDN model
"""
import collections
import scanpy.api as sc
import numpy as np
import tensorflow as tf
from sklearn import preprocessing as pp
import pandas as pd


def squeeze_fractions(features):
    """
    Custom activation functions that turns the logits into proportions
    that add up to 1 and are all positive
    :param features:
    :return:
    """
    abs_features = tf.abs(features)
    max = tf.reduce_sum(abs_features, axis=0)
    res = tf.divide(abs_features, max)
    return res


def _add_noise(x, y):
    """
    Add random noise to the sample x
    :param x: expression values
    :param y: ratios (not changed)
    :return: (x_noise, y)
    """
    # add random noise
    noise = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=0.2, dtype=tf.float32)
    x = x + noise
    # Scale again to [0,1] range
    mms = pp.MinMaxScaler(feature_range=(0, 1), copy=True)
    x = mms.fit_transform(x.T).T
    return x, y


def load_h5ad_file(input_path, batch_size, add_noise=False, datasets=['data8k', 'data6k', 'data68k', 'donorB', 'donorA', 'donorC', 'GSE65133']):
    """
    Load input data from a h5ad file and divide into training and test set
    :param input_path: path to h5ad file
    :param batch_size: batch size to use for training
    :param datasets: a list of datasets to extract from the file
    :return: Dataset object
    """
    raw_input = sc.read_h5ad(input_path)
    test_input = raw_input.copy()
    print(raw_input.shape)


    # divide dataset in train and test data
    all_ds = collections.Counter(raw_input.obs['ds'])
    for ds in all_ds:
        if ds in datasets:
            test_input = test_input[test_input.obs['ds'] != ds].copy()
        else:
            raw_input = raw_input[raw_input.obs['ds'] != ds].copy()

    # Create training dataset
    ratios = [raw_input.obs[ctype] for ctype in raw_input.uns['cell_types']]
    x_data = raw_input.X.astype(np.float32)
    y_data = np.array(ratios, dtype=np.float32).transpose()
    # create placeholders
    x_data_ph = tf.placeholder(x_data.dtype, x_data.shape)
    y_data_ph = tf.placeholder(y_data.dtype, y_data.shape)
    data = tf.data.Dataset.from_tensor_slices((x_data_ph, y_data_ph))
    # Add noise
    if add_noise:
        data = data.map(_add_noise)
    data_train = data.shuffle(1000).repeat().batch(batch_size=batch_size)

    # Create test dataset
    ratios = [test_input.obs[ctype] for ctype in test_input.uns['cell_types']]
    x_test = test_input.X.astype(np.float32)
    y_test = np.array(ratios, dtype=np.float32).transpose()
    x_test_ph = tf.placeholder(x_test.dtype, x_test.shape)
    y_test_ph = tf.placeholder(y_test.dtype, y_test.shape)
    data_test = tf.data.Dataset.from_tensor_slices((x_test_ph, y_test_ph))
    data_test = data_test.batch(batch_size=test_input.shape[0])
    # Extract info
    labels = raw_input.uns['cell_types']

    return data_train, (x_data_ph, y_data_ph), data_test, (x_test_ph, x_data_ph), (x_data, y_data), (x_test, y_test), labels

def dummy_labels(m, labels):
    """
    Create dummy labels needed for building the graph correctly
    :param m:
    :param labels:
    :return:
    """
    n_l = len(labels)
    return np.zeros((m, n_l), dtype="float32")


def load_prediction_file(input_path, sig_genes, labels, scaling=None):
    """
    Load a file to perform prediction on it
    :param input_path: path to input file
    :param sig_genes: the signature genes to use
    :param scaling: which scaling to perform
    :return: Dataset object
    """
    # Load data
    data = pd.read_table(input_path, sep="\t", index_col=0)
    sample_names = list(data.columns)
    data = data.loc[sig_genes]
    data = data.T
    data = data.astype(np.float32)
    m = data.shape[0]
    y_dummy = dummy_labels(m, labels)
    # Scaling
    if not scaling == None:
        data = sample_scaling(data, scaling_option=scaling)

    # Create Dataset object
    dataset = tf.data.Dataset.from_tensor_slices((data, y_dummy))
    dataset = dataset.batch(batch_size=m)
    return dataset, m, sample_names

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

    elif scaling_option == "min_max":
        # Normalize data
        mms = pp.MinMaxScaler(feature_range=(0, 1), copy=True)

        # it scales features so transpose is needed
        x = mms.fit_transform(x.T).T

    elif scaling_option == "log_robust":
        x = np.log2(x + 1)
        rs = pp.RobustScaler(copy=True)
        x = rs.fit_transform(x.T).T

    elif scaling_option == "robust":
        rs = pp.RobustScaler(copy=True)
        x = rs.fit_transform(x.T).T

    elif scaling_option == "quantile":
        qt = pp.QuantileTransformer()
        x = qt.fit_transform(x.T).T

    elif scaling_option == "log_quantile":
        x = np.log2(x + 1)
        qt = pp.QuantileTransformer()
        x = qt.fit_transform(x.T).T

    elif scaling_option == "quantile_gauss":
        qt = pp.QuantileTransformer(output_distribution="normal")
        x = qt.fit_transform(x.T).T

    elif scaling_option == "feat_log_min_max":
        x = np.log2(x + 1)
        mms = pp.MinMaxScaler(feature_range=(0,1))
        x = mms.fit_transform(x)

    elif scaling_option == "feat_quant_gauss":
        qt = pp.QuantileTransformer(output_distribution="normal")
        x = qt.fit_transform(x)

    elif scaling_option == "power_transform":
        pt = pp.PowerTransformer()
        x = pt.fit_transform(x.T).T

    elif scaling_option == "lsn":
        lib_size = 1000
        total = x.sum(axis=1)
        norm_factors = np.divide(lib_size, total)
        nf_array = np.array([norm_factors]*x.shape[1]).T
        print(nf_array.shape)
        x = np.multiply(x, nf_array)


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




