"""
Configuração avançada de verificação formal para experimentos Wine.
Inclui invariantes, pré/pós-condições e verificação de robustez.
"""

from typing import Dict, Any

# 🔒 CONFIGURAÇÃO DE INVARIANTES
WINE_INVARIANTS_CONFIG = {
    'invariant': {
        'invariants': [
            {
                'type': 'data_consistency',
                'description': 'Dados devem estar consistentes (sem NaN/Inf)'
            },
            {
                'type': 'model_stability', 
                'threshold': 0.1,
                'description': 'Modelo deve permanecer estável sob pequenas perturbações'
            },
            {
                'type': 'parameter_validity',
                'bounds': {
                    'C': {'min': 0.001, 'max': 1000.0},
                    'max_iter': {'min': 100, 'max': 2000},
                    'tol': {'min': 1e-8, 'max': 1e-2}
                },
                'description': 'Parâmetros devem estar em ranges válidos'
            }
        ]
    }
}

# 🔧 CONFIGURAÇÃO DE PRÉ-CONDIÇÕES  
WINE_PRECONDITIONS_CONFIG = {
    'precondition': {
        'conditions': [
            {
                'type': 'data_preprocessing',
                'description': 'Dados devem estar normalizados'
            },
            {
                'type': 'parameter_initialization',
                'required_params': ['C', 'max_iter', 'solver', 'random_state'],
                'description': 'Parâmetros obrigatórios devem estar inicializados'
            },
            {
                'type': 'data_shape_validation',
                'expected_shape': [36, 13],  # Wine dataset: 36 samples test, 13 features
                'description': 'Forma dos dados deve corresponder ao esperado'
            }
        ]
    }
}

# ⚡ CONFIGURAÇÃO DE PÓS-CONDIÇÕES
WINE_POSTCONDITIONS_CONFIG = {
    'postcondition': {
        'conditions': [
            {
                'type': 'output_validity',
                'description': 'Saída deve ser válida (sem NaN/Inf)'
            },
            {
                'type': 'probability_bounds',
                'description': 'Probabilidades devem estar entre 0 e 1'
            },
            {
                'type': 'classification_constraints',
                'num_classes': 3,  # Wine dataset tem 3 classes
                'description': 'Classes preditas devem estar no range [0, 2]'
            }
        ]
    }
}

# 🛡️ CONFIGURAÇÃO DE ROBUSTEZ
WINE_ROBUSTNESS_CONFIG = {
    'robustness': {
        'tests': [
            {
                'type': 'adversarial_robustness',
                'epsilon': 0.1,
                'norm': 'l2',
                'output_threshold': 0.1,
                'description': 'Resistência a ataques adversariais L2'
            },
            {
                'type': 'noise_robustness',
                'noise_level': 0.05,
                'stability_threshold': 0.05,
                'description': 'Resistência a ruído gaussiano'
            },
            {
                'type': 'parameter_sensitivity',
                'parameter_delta': 0.01,
                'output_delta_max': 0.1,
                'description': 'Sensibilidade a mudanças nos parâmetros'
            },
            {
                'type': 'distributional_robustness',
                'distribution_shift': 0.1,
                'performance_threshold': 0.9,
                'description': 'Robustez a mudanças na distribuição'
            }
        ]
    }
}

# 🎯 CONFIGURAÇÃO COMPLETA PARA WINE EXPERIMENTS
WINE_ADVANCED_VERIFICATION_CONFIG = {
    **WINE_INVARIANTS_CONFIG,
    **WINE_PRECONDITIONS_CONFIG, 
    **WINE_POSTCONDITIONS_CONFIG,
    **WINE_ROBUSTNESS_CONFIG,
    
    # Configurações básicas existentes
    'bounds': {
        'min': -1000.0,
        'max': 1000.0,
        'strict': False,
        'allow_nan': False
    },
    'range_check': {
        'type': 'continuous',
        'valid_ranges': [(-100.0, 100.0)],
        'allow_empty': False,
        'tolerance': 1e-06
    },
    'type_safety': True,
    'shape_preservation': True,
    'non_negative': True,
    'real_arithmetic': True
}

# 📊 CONFIGURAÇÕES ESPECÍFICAS POR ALGORITMO
LOGISTIC_REGRESSION_WINE_CONFIG = {
    **WINE_ADVANCED_VERIFICATION_CONFIG,
    'logistic_regression_convergence': {
        'max_iter': 1000,
        'tol': 1e-6
    },
    'logistic_regression_probability_bounds': True
}

MLP_WINE_CONFIG = {
    **WINE_ADVANCED_VERIFICATION_CONFIG,
    'mlp_architecture_validity': {
        'hidden_layer_sizes': (100,),
        'input_size': 13,
        'output_size': 3
    }
}

DECISION_TREE_WINE_CONFIG = {
    **WINE_ADVANCED_VERIFICATION_CONFIG,
    'decision_tree_purity': {
        'criterion': 'gini',
        'n_classes': 3
    }
}

def get_wine_verification_config(algorithm_name: str) -> Dict[str, Any]:
    """
    Retorna configuração de verificação específica para algoritmo no dataset Wine.
    
    Args:
        algorithm_name: Nome do algoritmo ('LogisticRegression', 'MLP', 'DecisionTree')
        
    Returns:
        Configuração de verificação formal
    """
    config_map = {
        'LogisticRegressionModel': LOGISTIC_REGRESSION_WINE_CONFIG,
        'LogisticRegression': LOGISTIC_REGRESSION_WINE_CONFIG,
        'MLPModel': MLP_WINE_CONFIG,
        'MLP': MLP_WINE_CONFIG,
        'DecisionTreeModel': DECISION_TREE_WINE_CONFIG,
        'DecisionTree': DECISION_TREE_WINE_CONFIG
    }
    
    return config_map.get(algorithm_name, WINE_ADVANCED_VERIFICATION_CONFIG)

def get_wine_verification_summary() -> Dict[str, Any]:
    """Retorna resumo das verificações implementadas para Wine."""
    return {
        'invariants': len(WINE_INVARIANTS_CONFIG['invariant']['invariants']),
        'preconditions': len(WINE_PRECONDITIONS_CONFIG['precondition']['conditions']),
        'postconditions': len(WINE_POSTCONDITIONS_CONFIG['postcondition']['conditions']),
        'robustness_tests': len(WINE_ROBUSTNESS_CONFIG['robustness']['tests']),
        'total_constraints': (
            len(WINE_INVARIANTS_CONFIG['invariant']['invariants']) +
            len(WINE_PRECONDITIONS_CONFIG['precondition']['conditions']) + 
            len(WINE_POSTCONDITIONS_CONFIG['postcondition']['conditions']) +
            len(WINE_ROBUSTNESS_CONFIG['robustness']['tests']) +
            6  # Constraints básicos
        ),
        'supported_algorithms': ['LogisticRegression', 'MLP', 'DecisionTree']
    }