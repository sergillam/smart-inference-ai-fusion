"""
🎯 Configuração Avançada de Verificação Formal Z3
Configurações específicas para máximo desempenho nos algoritmos e datasets escolhidos.
"""

# 🎯 ALGORITMOS ESPECÍFICOS DO EXPERIMENTO
LOGISTIC_REGRESSION_CONSTRAINTS = {
    'logistic_regression_convergence': {
        'max_iter': 1000,
        'tol': 1e-6,
        'description': 'Verificar convergência da regressão logística'
    },
    'logistic_regression_probability_bounds': {
        'min_prob': 0.0001,
        'max_prob': 0.9999,
        'description': 'Verificar bounds de probabilidade sigmoid'
    },
    'bounds': {
        'min': -100.0,
        'max': 100.0,
        'strict': False,
        'allow_nan': False,
        'description': 'Bounds para coeficientes LogisticRegression'
    },
    'range_check': {
        'type': 'continuous',
        'valid_ranges': [(-10.0, 10.0)],
        'tolerance': 1e-9,
        'description': 'Range check para estabilidade numérica'
    }
}

DECISION_TREE_CONSTRAINTS = {
    'decision_tree_purity': {
        'criterion': 'gini',
        'n_classes': 3,
        'min_impurity_decrease': 0.0,
        'description': 'Verificar pureza dos nós da árvore'
    },
    'bounds': {
        'min': 0,
        'max': 1000,
        'strict': False,
        'allow_nan': False,
        'description': 'Bounds para contagem de amostras por nó'
    },
    'integer_arithmetic': {
        'description': 'Verificar aritmética inteira para contagens'
    },
    'range_check': {
        'type': 'discrete',
        'discrete_values': list(range(1000)),
        'allow_empty': False,
        'description': 'Range check para índices de features'
    }
}

MLP_CLASSIFIER_CONSTRAINTS = {
    'mlp_architecture_validity': {
        'hidden_layer_sizes': (100, 50),
        'input_size': 13,  # Wine dataset features
        'output_size': 3,   # Wine classes
        'description': 'Verificar arquitetura válida da rede neural'
    },
    'bounds': {
        'min': -10.0,
        'max': 10.0,
        'strict': False,
        'allow_nan': False,
        'description': 'Bounds para pesos da rede neural'
    },
    'real_arithmetic': {
        'description': 'Verificar aritmética real para computações MLP'
    },
    'range_check': {
        'type': 'continuous',
        'valid_ranges': [(-1.0, 1.0)],  # Para ativações normalizadas
        'tolerance': 1e-6,
        'description': 'Range check para ativações'
    }
}

# 📊 DATASET-SPECIFIC CONSTRAINTS
ADULT_DATASET_CONSTRAINTS = {
    'adult_fairness_constraints': {
        'protected_attributes': ['age', 'sex', 'race'],
        'fairness_metric': 'demographic_parity',
        'tolerance': 0.1,
        'description': 'Verificar fairness para previsão de renda'
    },
    'bounds': {
        'min': 0,
        'max': 99,
        'strict': False,
        'allow_nan': False,
        'description': 'Bounds para idade no dataset Adult'
    }
}

BREAST_CANCER_CONSTRAINTS = {
    'bounds': {
        'min': 0.0,
        'max': 1000.0,
        'strict': False,
        'allow_nan': False,
        'description': 'Bounds para features médicas'
    },
    'range_check': {
        'type': 'continuous',
        'valid_ranges': [(0.0, 50.0), (50.0, 200.0), (200.0, 1000.0)],
        'tolerance': 1e-6,
        'description': 'Ranges médicos válidos'
    },
    'non_negative': {
        'description': 'Features médicas devem ser não-negativas'
    }
}

WINE_DATASET_CONSTRAINTS = {
    'wine_quality_classification': {
        'n_features': 13,
        'n_classes': 3,
        'feature_names': ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash',
                         'magnesium', 'total_phenols', 'flavanoids', 
                         'nonflavanoid_phenols', 'proanthocyanins',
                         'color_intensity', 'hue', 'od280_od315_of_diluted_wines', 'proline'],
        'description': 'Verificar classificação de qualidade do vinho'
    },
    'bounds': {
        'min': 0.0,
        'max': 100.0,
        'strict': False,
        'allow_nan': False,
        'description': 'Bounds para features químicas do vinho'
    },
    'range_check': {
        'type': 'continuous',
        'valid_ranges': [(0.0, 20.0)],  # Valores típicos para wine features
        'tolerance': 1e-6,
        'description': 'Range check para propriedades químicas'
    }
}

MAKE_MOONS_CONSTRAINTS = {
    'make_moons_separability': {
        'n_samples': 1000,
        'noise': 0.1,
        'random_state': 42,
        'description': 'Verificar separabilidade para dataset sintético'
    },
    'bounds': {
        'min': -3.0,
        'max': 4.0,
        'strict': False,
        'allow_nan': False,
        'description': 'Bounds para coordenadas make_moons'
    },
    'range_check': {
        'type': 'continuous',
        'valid_ranges': [(-2.5, 3.5), (-1.5, 2.5)],  # x1, x2 ranges
        'tolerance': 1e-6,
        'description': 'Range check para coordenadas 2D'
    }
}

# 🚀 CONFIGURAÇÕES DE EXPERIMENTO COMPLETAS
EXPERIMENT_CONFIGS = {
    'logistic_regression_adult': {
        'algorithm_constraints': LOGISTIC_REGRESSION_CONSTRAINTS,
        'dataset_constraints': ADULT_DATASET_CONSTRAINTS,
        'description': 'LogisticRegression no dataset Adult com verificação de fairness'
    },
    'logistic_regression_breast_cancer': {
        'algorithm_constraints': LOGISTIC_REGRESSION_CONSTRAINTS,
        'dataset_constraints': BREAST_CANCER_CONSTRAINTS,
        'description': 'LogisticRegression no dataset Breast Cancer com constraints médicos'
    },
    'decision_tree_wine': {
        'algorithm_constraints': DECISION_TREE_CONSTRAINTS,
        'dataset_constraints': WINE_DATASET_CONSTRAINTS,
        'description': 'DecisionTree no dataset Wine com verificação de pureza'
    },
    'decision_tree_make_moons': {
        'algorithm_constraints': DECISION_TREE_CONSTRAINTS,
        'dataset_constraints': MAKE_MOONS_CONSTRAINTS,
        'description': 'DecisionTree no make_moons com análise de separabilidade'
    },
    'mlp_wine': {
        'algorithm_constraints': MLP_CLASSIFIER_CONSTRAINTS,
        'dataset_constraints': WINE_DATASET_CONSTRAINTS,
        'description': 'MLPClassifier no dataset Wine (atual experimento)'
    },
    'mlp_make_moons': {
        'algorithm_constraints': MLP_CLASSIFIER_CONSTRAINTS,
        'dataset_constraints': MAKE_MOONS_CONSTRAINTS,
        'description': 'MLPClassifier no make_moons com boundaries não-lineares'
    }
}

# 🎯 CONFIGURAÇÃO DE MÁXIMO DESEMPENHO
MAX_PERFORMANCE_CONFIG = {
    'z3_timeout': 300000,  # 5 minutos por constraint
    'z3_threads': 16,      # Usar todos os cores
    'z3_memory_limit': 12000,  # 12GB para Z3
    'parallel_verification': True,
    'aggressive_preprocessing': True,
    'detailed_counterexamples': True,
    'statistical_analysis': True,
    'performance_profiling': True
}

def get_experiment_config(experiment_name: str) -> dict:
    """Retorna configuração completa para um experimento específico."""
    if experiment_name not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Experimento '{experiment_name}' não encontrado. Disponíveis: {list(EXPERIMENT_CONFIGS.keys())}")
    
    config = EXPERIMENT_CONFIGS[experiment_name].copy()
    config['performance'] = MAX_PERFORMANCE_CONFIG
    return config

def get_all_constraints_for_experiment(experiment_name: str) -> dict:
    """Retorna todos os constraints consolidados para um experimento."""
    config = get_experiment_config(experiment_name)
    
    all_constraints = {}
    all_constraints.update(config['algorithm_constraints'])
    all_constraints.update(config['dataset_constraints'])
    
    return all_constraints