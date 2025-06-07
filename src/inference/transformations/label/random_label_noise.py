import numpy as np
from inference.transformations.label.base import LabelTransformation

class RandomLabelNoise(LabelTransformation):
    def __init__(self, flip_fraction):
        self.flip_fraction = flip_fraction

    def apply(self, y):
        y_np = np.asarray(y)  # garante indexação posicional
        y_noisy = y_np.copy()
        n = int(len(y_np) * self.flip_fraction)
        indices = np.random.choice(len(y_np), n, replace=False)
        classes = np.unique(y_np)
        for i in indices:
            available = classes[classes != y_noisy[i]]
            y_noisy[i] = np.random.choice(available)
        return y_noisy