"""
Generate random example data which allows for testing and
to give users examples for the input format
"""

import pandas as pd
import numpy as np
import string
import random


def exampleCounts(n_cells=10, n_genes=100):
    """
    Generate an example scRNA-seq count file
    :param n: number of cells
    :param g: number of genes
    """
    counts = np.random.randn(n_cells, n_genes)
    gene_name_lengths = np.random.choice(list(range(3, 10)), n_genes)
    gene_names = []
    for gl in gene_name_lengths:
        gene_names.append(''.join(random.choices(string.ascii_letters, k=gl)))

    df = pd.DataFrame(counts, columns=gene_names)
    print(df)
