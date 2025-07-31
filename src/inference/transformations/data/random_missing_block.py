"""Transformation for simulating random missing blocks in input data."""

import numpy as np
from inference.transformations.data.base import InferenceTransformation

class RandomMissingBlock(InferenceTransformation):
    """Simulates sensor or data failure by randomly setting blocks to NaN or zero.

    This transformation selects random contiguous blocks (rectangles) of the input data
    and sets their values to NaN or zero, imitating block-wise sensor faults.

    Attributes:
        block_fraction (float): Proportion of the dataset to corrupt (0.0 to 1.0).
        zero_fill (bool): If True, fills with 0 instead of NaN.
    """

    def __init__(self, block_fraction=0.1, zero_fill=False):
        """Initializes the random missing block transformation.

        Args:
            block_fraction (float): Proportion of the dataset to corrupt (0.0 to 1.0).
            zero_fill (bool): If True, fills blocks with 0.0 instead of NaN.
        """
        self.block_fraction = block_fraction
        self.zero_fill = zero_fill

    def apply(self, X):
        """Applies block-wise missing/corrupt values to input data.

        Args:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Perturbed feature matrix with missing or zero blocks.
        """
        x_perturbed = X.copy()
        n_rows, n_cols = x_perturbed.shape
        total_cells = int(n_rows * n_cols * self.block_fraction)

        count = 0
        while count < total_cells:
            block_rows = np.random.randint(1, min(5, n_rows))
            block_cols = np.random.randint(1, min(5, n_cols))
            start_row = np.random.randint(0, n_rows - block_rows + 1)
            start_col = np.random.randint(0, n_cols - block_cols + 1)

            value = 0.0 if self.zero_fill else np.nan
            x_perturbed[start_row:start_row + block_rows, start_col:start_col + block_cols] = value

            count += block_rows * block_cols

        return x_perturbed
