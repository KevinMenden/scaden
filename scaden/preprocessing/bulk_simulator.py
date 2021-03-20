class BulkSimulator(object):
    def __init__(self, sample_size, num_samples, data_path, out_dir, pattern, unknown_celltypes):
        self.sample_size = sample_size
        self.num_samples = num_samples
        self.data_path = data_path
        self.out_dir = out_dir
        self.pattern = pattern
        self.unknown_celltypes = unknown_celltypes

