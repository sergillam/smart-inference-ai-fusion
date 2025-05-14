from inference.transformations.params.parameter_perturber import SmartParameterPerturber

def apply_param_inference(
    model_class,
    *,
    base_params=None,
    seed=None,
    ignore_rules=None
):
    """
    Aplica inferência nos hiperparâmetros de um modelo de IA.

    Utiliza a classe SmartParameterPerturber para alterar dinamicamente
    os parâmetros com base em regras e tipo de dado.

    Args:
        model_class (Callable): Classe do modelo (ex: KNNModel).
        base_params (dict, optional): Parâmetros base originais do modelo.
        seed (int, optional): Semente para reprodutibilidade das perturbações.
        ignore_rules (set[str], optional): Conjunto de nomes de parâmetros a serem ignorados.

    Returns:
        Tuple:
            model: Instância do modelo com parâmetros perturbados.
            log (dict): Dicionário com o histórico das alterações aplicadas.
    """
    perturber = SmartParameterPerturber(base_params, seed=seed, ignore_rules=ignore_rules)
    perturbed_params = perturber.apply(model_class=model_class)
    model = model_class(**perturbed_params)
    log = perturber.export_log()
    return model, log
