"""
Generate random example data which allows for testing and
to give users examples for the input format
"""
import string
import random
import os
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def exampleData(n_cells=10, n_genes=100, n_samples=10, out_dir="./"):
    """
    Generate an example scRNA-seq count file
    :param n: number of cells
    :param g: number of genes
    """

    # Generate example scRNA-seq data
    counts = np.random.randint(low=0, high=1000, size=(n_cells, n_genes))
    gene_names = ['gene'] * n_genes
    for i in range(len(gene_names)):
        gene_names[i] = gene_names[i] + str(i)
    df = pd.DataFrame(counts, columns=gene_names)

    # Generate example celltype labels
    celltypes = ['celltype'] * np.random.randint(low=2, high=n_cells - 1)
    for i in range(len(celltypes)):
        celltypes[i] = celltypes[i] + str(i)
    celltype_list = random.choices(celltypes, k=n_cells)
    ct_df = pd.DataFrame(celltype_list, columns=['Celltype'])

    # Generate example bulk RNA-seq data
    bulk = np.random.randint(low=0, high=1000, size=(n_genes, n_samples))
    samples = ['sample'] * n_samples
    for i in range(len(samples)):
        samples[i] = samples[i] + str(i)
    bulk_df = pd.DataFrame(bulk, columns=samples, index=gene_names)

    # Save the data
    df.to_csv(os.path.join(out_dir, "example_counts.txt"), sep="\t")
    ct_df.to_csv(os.path.join(out_dir, "example_celltypes.txt"),
                 sep="\t",
                 index=False)
    bulk_df.to_csv(os.path.join(out_dir, "example_bulk_data.txt"), sep="\t")

    logger.warn(f"Example data has been created in {out_dir}")
