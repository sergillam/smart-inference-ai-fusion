from inference.parameter_perturber import SmartParameterPerturber

def apply_param_inference(model_class, *, base_params=None, seed=None, ignore_rules=None):
    """
    Aplica inferência aos parâmetros do modelo usando SmartParameterPerturber.

    Args:
        model_class: Classe do modelo (ex: KNNModel)
        base_params: Dicionário com os hiperparâmetros originais
        seed: Semente para reprodutibilidade
        ignore_rules: Dicionário com parâmetros a ignorar nas perturbações

    Returns:
        Tuple: (instância do modelo com parâmetros perturbados, log das alterações)
    """
    perturber = SmartParameterPerturber(base_params, seed=seed, ignore_rules=ignore_rules)
    perturbed_params = perturber.apply(model_class=model_class)
    model = model_class(**perturbed_params)
    log = perturber.export_log()
    return model, log
