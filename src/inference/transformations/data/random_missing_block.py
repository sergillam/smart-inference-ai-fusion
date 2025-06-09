import numpy as np
from inference.transformations.data.base import InferenceTransformation

class RandomMissingBlock(InferenceTransformation):
    """
    Simulates a block-wise sensor/data failure by randomly selecting contiguous
    blocks of rows and columns and setting them to NaN or zero.
    """

    def __init__(self, block_fraction=0.1, zero_fill=False):
        """
        Args:
            block_fraction (float): Proportion of the dataset to corrupt (0.0 to 1.0).
            zero_fill (bool): If True, fills with 0 instead of NaN.
        """
        self.block_fraction = block_fraction
        self.zero_fill = zero_fill

    def apply(self, X):
        X_perturbed = X.copy()
        n_rows, n_cols = X_perturbed.shape
        total_cells = int(n_rows * n_cols * self.block_fraction)

        # Estimating number of blocks to place
        avg_block_size = max(1, total_cells // 10)  # target ~10 blocks

        count = 0
        while count < total_cells:
            block_rows = np.random.randint(1, min(5, n_rows))
            block_cols = np.random.randint(1, min(5, n_cols))
            start_row = np.random.randint(0, n_rows - block_rows + 1)
            start_col = np.random.randint(0, n_cols - block_cols + 1)

            value = 0.0 if self.zero_fill else np.nan
            X_perturbed[start_row:start_row+block_rows, start_col:start_col+block_cols] = value

            count += block_rows * block_cols

        return X_perturbed
