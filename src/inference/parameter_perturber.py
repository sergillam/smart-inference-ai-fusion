import random
import copy

class SmartParameterPerturber:
    def __init__(self, params: dict, seed: int = None, ignore_rules: dict = None):
        self.original_params = params
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self.ignore_rules = ignore_rules or {}
        self.perturbation_log = {}
        self.memo = set()
        self.forbidden_names = ['C', 'kernel']  # Exemplo de parâmetros obrigatórios

    def apply(self, model_class=None):
        new_params = copy.deepcopy(self.original_params)
        by_type = self._group_by_type(new_params)

        for ptype, keys in by_type.items():
            if not keys:
                continue
            key = random.choice(keys)
            if key in self.ignore_rules or key in self.forbidden_names:
                continue
            original = new_params[key]
            perturbed, strategy = self._perturb(key, original, ptype)
            if perturbed is not None:
                new_params[key] = perturbed
                self.perturbation_log[key] = {
                    "original": original,
                    "perturbed": perturbed,
                    "strategy": strategy
                }

        # Fallback test
        if model_class:
            try:
                model_class(**new_params)
            except Exception:
                return self.original_params  # fallback

        return new_params

    def _group_by_type(self, params):
        type_groups = {int: [], float: [], str: [], bool: [], type(None): []}
        for k, v in params.items():
            type_groups.get(type(v), []).append(k)
        return type_groups

    def _perturb(self, key, value, ptype):
        if ptype == int:
            return self._perturb_int(value)
        elif ptype == float:
            return self._perturb_float(value)
        elif ptype == str:
            return self._perturb_str(value)
        elif ptype == bool:
            return self._perturb_bool(value)
        elif ptype == type(None):
            return 0, "replace_none"
        return None, "unsupported_type"

    def _perturb_int(self, val):
        strategy = random.choice(["add_noise", "cast_str", "cast_float", "drop"])
        try:
            if strategy == "add_noise":
                return val + random.randint(-2, 2), strategy
            elif strategy == "cast_str":
                return str(val), strategy
            elif strategy == "cast_float":
                return float(val), strategy
            elif strategy == "drop":
                return None, strategy
        except Exception:
            return val, "fallback"
        return val, "noop"

    def _perturb_float(self, val):
        strategy = random.choice(["add_noise", "cast_str", "drop"])
        try:
            if strategy == "add_noise":
                return val + random.uniform(-0.5, 0.5), strategy
            elif strategy == "cast_str":
                return str(val), strategy
            elif strategy == "drop":
                return None, strategy
        except Exception:
            return val, "fallback"
        return val, "noop"

    def _perturb_str(self, val):
        fake_options = ["invalid", "none", "off", "unknown"]
        strategy = random.choice(["mutate", "drop", "cast"])
        try:
            if strategy == "mutate":
                return random.choice(fake_options), strategy
            elif strategy == "drop":
                return None, strategy
            elif strategy == "cast":
                return str(val) + "_x", strategy
        except Exception:
            return val, "fallback"
        return val, "noop"

    def _perturb_bool(self, val):
        strategy = random.choice(["flip", "cast_str", "drop"])
        try:
            if strategy == "flip":
                return not val, strategy
            elif strategy == "cast_str":
                return str(val), strategy
            elif strategy == "drop":
                return None, strategy
        except Exception:
            return val, "fallback"
        return val, "noop"

    def export_log(self):
        return self.perturbation_log