"""
Combine artificial bulk datasets and optionally other datasets into h5ad files
usable for scaden training

When using additional datasets, they should be in similar format and best have the same output cell types.
"""
import gc
import anndata
import glob
import os
import sys
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def parse_data(x_path, y_path):
    """
    Parse data and labels and divide them into training and testset
    :param x_path:
    :param y_path:
    :return: training and test data and labels
    """
    # Load the data
    try:
        x = pd.read_table(x_path, sep="\t")
        y = pd.read_table(y_path, sep="\t")
    except FileNotFoundError as e:
        logging.error(f"   Could not find simulated data files: {e}")
        sys.exit()
    labels = list(y.columns)

    # Transform Y to numpy array and split in train and testset
    yseries = []

    for i in range(y.shape[0]):
        yseries.append(list(y.iloc[i]))
    y = np.array(yseries)

    return x, y, labels


def load_celltypes(brain_training_dir):

    celltypes_path = os.path.join(brain_training_dir, "celltypes.txt")
    celltypes = pd.read_csv(celltypes_path, sep="\t")
    celltypes = list(celltypes.iloc[:, 1])
    return celltypes


def sort_celltypes(ratios, labels, ref_labels):
    """
    Bring ratios in correct order of cell types
    :param ratios:
    :param labels:
    :param ref_labels:
    :return:
    """
    idx = [labels.index(x) for x in ref_labels]
    ratios = ratios[:, idx]
    return ratios


def create_h5ad_file(data_dir, out_path, unknown, pattern="*_samples.txt"):
    """
    Create h5ad file from simulated data
    """

    # List available datasets
    files = glob.glob(data_dir + pattern)
    files = [os.path.basename(x) for x in files]
    datasets = [x.replace(pattern.replace("*", ""), "") for x in files]

    # get celltypes
    celltypes = load_celltypes(data_dir)
    print(f"Celltypes: {celltypes}")
    print(f"Found datasets: {datasets}")
    adata = []
    me_dict = {}

    # Create adata datasets for each
    for i, train_file in enumerate(datasets):

        rna_file = os.path.join(data_dir, train_file + "_samples.txt")
        ratios_file = os.path.join(data_dir, train_file + "_labels.txt")

        x, y, labels = parse_data(rna_file, ratios_file)

        # sort y
        y = sort_celltypes(y, labels, celltypes)
        test = [labels.index(x) for x in celltypes]
        labels = [labels[i] for i in test]

        x = x.sort_index(axis=1)
        ratios = pd.DataFrame(y, columns=celltypes)
        ratios["ds"] = pd.Series(np.repeat(train_file, y.shape[0]), index=ratios.index)

        print("Processing " + str(train_file))
        x = pd.DataFrame(x)
        adata.append(
            anndata.AnnData(
                X=x.to_numpy(), obs=ratios, var=pd.DataFrame(columns=[], index=list(x))
            )
        )

    for i in range(1, len(adata)):
        print("Concatenating " + str(i))
        adata[0] = adata[0].concatenate(adata[1])
        del adata[1]
        gc.collect()
        print(len(adata))
    adata = adata[0]

    # add cell types and signature genes
    adata.uns["cell_types"] = celltypes
    adata.uns["unknown"] = unknown

    # save data
    adata.write(out_path)
