from .base import ParameterTransformation

class CrossDependencyPerturbation(ParameterTransformation):
    """
    Alters dependent parameters together based on predefined rules.
    For example: if 'penalty' = 'l1', enforce 'solver' = 'liblinear'.
    """

    RULES = [
        {
            "condition": {"penalty": "l1"},
            "set": {"solver": "liblinear"},
        },
        {
            "condition": {"penalty": "l2"},
            "set": {"solver": "lbfgs"},
        },
        {
            "condition": {"kernel": "linear"},
            "set": {"gamma": "auto"},
        },
        # Add more rules as needed
    ]

    def apply(self, params: dict) -> None:
        for rule in self.RULES:
            condition = rule["condition"]
            if all(params.get(k) == v for k, v in condition.items()):
                for target_key, target_value in rule["set"].items():
                    if params.get(target_key) != target_value:
                        params[target_key] = target_value  # Apply directly
        return None  # No need to return a specific value

    @staticmethod
    def supports(value) -> bool:
        # Applies to any type — this perturber checks full param context
        return True
