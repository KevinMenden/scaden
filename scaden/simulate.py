from scaden.preprocessing.create_h5ad_file import create_h5ad_file
from scaden.preprocessing.bulk_simulation import simulate_bulk
from scaden.preprocessing import BulkSimulator
import os
"""
Simulation of artificial bulk RNA-seq samples from scRNA-seq data
and subsequenbt formatting in .h5ad file for training with Scaden
"""


def simulation(simulate_dir, data_dir, sample_size, num_samples, pattern,
               unknown_celltypes, out_prefix, fmt):

    unknown_celltypes = list(unknown_celltypes)
    bulk_simulator = BulkSimulator(sample_size=sample_size,
                                   num_samples=num_samples,
                                   data_path=data_dir,
                                   out_dir=simulate_dir,
                                   pattern=pattern,
                                   unknown_celltypes=unknown_celltypes,
                                   fmt=fmt)

    bulk_simulator.simulate()

    # Perform the bulk simulation
    #simulate_bulk(sample_size, num_samples, data_dir, simulate_dir, pattern,
    #              unknown_celltypes, fmt)

    # Create the h5ad training data file
    out_name = os.path.join(simulate_dir, out_prefix + ".h5ad")
    create_h5ad_file(simulate_dir, out_name, unknown_celltypes)
