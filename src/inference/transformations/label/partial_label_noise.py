import numpy as np
from inference.transformations.label.base import LabelTransformation

class PartialLabelNoise(LabelTransformation):
    """
    Simula ambiguidade nos rótulos: substitui uma fração deles por um rótulo alternativo válido.

    Ao invés de retornar "multi-rótulos" como string (ex: '3|0'), 
    escolhe aleatoriamente entre o verdadeiro e o alternativo para manter o tipo original.
    """

    def __init__(self, noise_fraction: float):
        self.noise_fraction = noise_fraction

    def apply(self, y):
        y_np = np.asarray(y)
        y_noisy = y_np.copy()

        n = int(len(y_np) * self.noise_fraction)
        indices = np.random.choice(len(y_np), n, replace=False)
        classes = np.unique(y_np)

        for idx in indices:
            true_label = y_noisy[idx]
            alternatives = classes[classes != true_label]
            alt_label = np.random.choice(alternatives)

            # Simula ambiguidade escolhendo aleatoriamente entre o verdadeiro e o alternativo
            y_noisy[idx] = np.random.choice([true_label, alt_label])

        return y_noisy
