"""Label transformation that swaps labels within the same superclass."""

from collections import defaultdict
import numpy as np
from inference.transformations.label.base import LabelTransformation

class LabelSwapWithinClass(LabelTransformation):
    """Swaps labels within samples belonging to the same superclass.

    Assumes label strings share a common prefix (superclass) separated by a delimiter
    (e.g., 'benign_1' and 'benign_2' both have superclass 'benign').

    Attributes:
        swap_within_class_fraction (float): Fraction of samples to swap within superclass.
        delimiter (str): Delimiter used to extract superclass from labels.
    """

    def __init__(self, swap_within_class_fraction: float, delimiter: str = "_"):
        """Initializes the LabelSwapWithinClass transformation.

        Args:
            swap_within_class_fraction (float): Fraction of labels to swap within each superclass.
            delimiter (str, optional): Delimiter used to identify superclass (default: "_").
        """
        self.swap_within_class_fraction = swap_within_class_fraction
        self.delimiter = delimiter

    def apply(self, y):
        """Swaps pairs of labels within each superclass, up to a given fraction.

        Args:
            y (array-like): Input label vector (converted to str internally).

        Returns:
            np.ndarray: Label vector with swapped labels within superclass.
        """
        y = np.asarray(y).astype(str)
        y_noisy = y.copy()
        n_samples = len(y)
        n_swaps = int(n_samples * self.swap_within_class_fraction)
        if n_swaps < 2:
            return y

        # Group sample indices by superclass
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
