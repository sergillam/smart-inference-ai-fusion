class LabelTransformation:
    """Classe base para técnicas de inferência aplicadas aos rótulos (y)."""

    def apply(self, y):
        """
        Aplica a transformação no vetor de rótulos y.

        Args:
            y (array-like): Rótulos originais.

        Returns:
            array-like: Rótulos transformados.
        """
        raise NotImplementedError("O método 'apply' deve ser implementado pela subclasse.")
