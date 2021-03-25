from scaden.simulation import BulkSimulator

"""
Merge simulate datasets
"""


def merge_datasets(data_dir, prefix, files=None):

    bulk_simulator = BulkSimulator()

    if files:
        files = files.split(",")

    # Merge the resulting datasets
    bulk_simulator.merge_datasets(data_dir=data_dir,
                                  files=files,
                                  out_name=prefix + ".h5ad")
