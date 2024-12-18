import numpy as np
import itertools as it


# minimalist grid sampler
class GridSampler:
    def __init__(self, sample_space):
        self.fixed = {k: v for k, v in sample_space.items() if np.isscalar(v)}
        
        varying = {k: v for k, v in sample_space.items() if k not in self.fixed}
        samples = zip(*list(it.product(*varying.values())))
        self.samples = {k: np.array(v) for k, v in zip(varying.keys(), samples)}

    def __getitem__(self, index):
        sample = {k: v[index] for k, v in self.samples.items()}
        sample.update(self.fixed)
        return sample