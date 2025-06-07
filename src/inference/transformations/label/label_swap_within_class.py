import numpy as np
from inference.transformations.label.base import LabelTransformation
from collections import defaultdict

class LabelSwapWithinClass(LabelTransformation):
    """
    Troca rótulos entre amostras de um mesmo supergrupo (ex: "benign1", "benign2").
    Assume que os rótulos compartilham um prefixo comum separável por delimitador.
    """

    def __init__(self, swap_within_class_fraction: float, delimiter: str = "_"):
        """
        Args:
            swap_within_class_fraction (float): Fração dos rótulos que serão trocados dentro da mesma superclasse.
            delimiter (str): Delimitador usado para identificar o supergrupo nos rótulos (ex: "benign_1").
        """
        self.swap_within_class_fraction = swap_within_class_fraction
        self.delimiter = delimiter

    def apply(self, y):
        y = np.asarray(y).astype(str)
        y_noisy = y.copy()
        n_samples = len(y)
        n_swaps = int(n_samples * self.swap_within_class_fraction)
        if n_swaps < 2:
            return y

        # Agrupa os índices por superclasse
        groups = defaultdict(list)
        for idx, label in enumerate(y):
            superclass = label.split(self.delimiter)[0]
            groups[superclass].append(idx)

        swaps_done = 0
        for indices in groups.values():
            if len(indices) < 2:
                continue
            np.random.shuffle(indices)
            for i in range(0, len(indices) - 1, 2):
                if swaps_done >= n_swaps:
                    break
                idx1, idx2 = indices[i], indices[i+1]
                y_noisy[idx1], y_noisy[idx2] = y_noisy[idx2], y_noisy[idx1]
                swaps_done += 2

        return y_noisy
