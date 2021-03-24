from scaden.simulation import BulkSimulator

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

    # Perform dataset simulation
    bulk_simulator.simulate()

    # Merge the resulting datasets
    bulk_simulator.merge_datasets(data_dir=simulate_dir,
                                  files=bulk_simulator.dataset_files,
                                  out_name=out_prefix + ".h5ad")
