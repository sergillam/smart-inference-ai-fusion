import numpy as np
from inference.transformations.label.base import LabelTransformation

class LabelNoise(LabelTransformation):
    def __init__(self, flip_fraction):
        self.flip_fraction = flip_fraction

    def apply(self, y):
        y_noisy = y.copy()
        n = int(len(y) * self.flip_fraction)
        indices = np.random.choice(len(y), n, replace=False)
        classes = np.unique(y)
        for i in indices:
            available = classes[classes != y[i]]
            y_noisy[i] = np.random.choice(available)
        return y_noisy