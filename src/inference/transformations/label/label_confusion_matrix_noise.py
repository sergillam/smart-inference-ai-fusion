import numpy as np
from inference.transformations.label.base import LabelTransformation

class LabelConfusionMatrixNoise(LabelTransformation):
    """
    Aplica ruído nos rótulos com base em uma matriz de confusão artificial.
    Classes são trocadas conforme probabilidades definidas entre pares de classes.
    """

    def __init__(self, noise_level: float):
        """
        Args:
            noise_level (float): Fração dos rótulos a serem alterados com base na matriz.
        """
        self.noise_level = noise_level

    def apply(self, y):
        y = np.asarray(y)
        y_noisy = y.copy()
        n_samples = len(y)
        n_flip = int(n_samples * self.noise_level)
        if n_flip == 0:
            return y

        classes = np.unique(y)
        n_classes = len(classes)

        # Criar matriz de confusão artificial (com maior probabilidade fora da diagonal)
        confusion_matrix = np.full((n_classes, n_classes), 1.0 / (n_classes - 1))
        np.fill_diagonal(confusion_matrix, 0)

        # Normalizar por linha (cada linha soma 1)
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        confusion_matrix /= row_sums

        indices = np.random.choice(n_samples, size=n_flip, replace=False)

        for idx in indices:
            current_class = y_noisy[idx]
            current_idx = np.where(classes == current_class)[0][0]

            # Sorteia nova classe com base na probabilidade da linha
            new_class = np.random.choice(classes, p=confusion_matrix[current_idx])
            y_noisy[idx] = new_class

        return y_noisy
