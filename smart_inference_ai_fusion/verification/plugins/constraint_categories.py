"""Shared constraint category utilities for verification plugins.

This module contains common functionality used by both Z3 and CVC5 plugins
to categorize and format constraints and counterexamples.
"""

from typing import Any, Dict

# Mapeamento de constraints para categorias com emojis
CONSTRAINT_CATEGORIES = {
    # Constraints de dados
    "bounds": "📊 Limites de Dados",
    "range_check": "📊 Verificação de Range",
    "non_negative": "📊 Não-Negatividade",
    "type_safety": "📊 Segurança de Tipos",
    # Aritmética
    "linear_arithmetic": "🔢 Aritmética Linear",
    "real_arithmetic": "🔢 Aritmética Real",
    "integer_arithmetic": "🔢 Aritmética Inteira",
    "floating_point": "🔢 Ponto Flutuante",
    "bitvector_arithmetic": "🔢 Bit-Vectors",
    # Estrutural
    "shape_preservation": "📐 Preservação de Shape",
    "array_theory": "📐 Teoria de Arrays",
    "string_theory": "📐 Teoria de Strings",
    # Contratos
    "invariant": "📜 Invariantes",
    "precondition": "📜 Pré-condições",
    "postcondition": "📜 Pós-condições",
    # ML/IA
    "robustness": "🤖 Robustez de ML",
    "neural_network": "🤖 Rede Neural",
    "probability_bounds": "🤖 Limites Probabilísticos",
    # Lógica
    "boolean_logic": "🔀 Lógica Booleana",
    "quantified_formulas": "🔀 Fórmulas Quantificadas",
    "optimization": "🔀 Otimização",
    # Parâmetros
    "parameter_drift": "⚙️ Drift de Parâmetros",
    "parameter_consistency": "⚙️ Consistência de Parâmetros",
    "model_instantiation": "⚙️ Instanciação de Modelo",
    "attribute_check": "⚙️ Verificação de Atributos",
}


def get_constraint_category(constraint_type: str) -> str:
    """Retorna a categoria de um constraint para melhor organização dos logs.

    Args:
        constraint_type: O tipo de constraint.

    Returns:
        A categoria formatada com emoji.
    """
    return CONSTRAINT_CATEGORIES.get(constraint_type, "❓ Outro")


def format_counterexamples_summary(
    constraints_violated: list, solver_details: dict
) -> Dict[str, Any]:
    """Formata um resumo estruturado dos contraexemplos para o relatório JSON.

    Args:
        constraints_violated: Lista de constraints que foram violados.
        solver_details: Dicionário com detalhes de cada constraint.

    Returns:
        Dicionário com resumo estruturado dos contraexemplos.
    """
    summary: Dict[str, Any] = {
        "total_violations": len(constraints_violated),
        "by_category": {},
        "by_constraint": {},
    }

    for constraint in constraints_violated:
        category = get_constraint_category(constraint)
        details = solver_details.get(constraint, {})
        counterexample = details.get("counterexample", {})

        # Agrupar por categoria
        if category not in summary["by_category"]:
            summary["by_category"][category] = []
        summary["by_category"][category].append(constraint)

        # Detalhes por constraint
        summary["by_constraint"][constraint] = {
            "category": category,
            "violation_count": counterexample.get("violation_count", 0),
            "has_counterexample": bool(counterexample),
            "example_types": list(
                set(
                    ex.get("type", "unknown") for ex in counterexample.get("violation_examples", [])
                )
            ),
        }

    return summary
