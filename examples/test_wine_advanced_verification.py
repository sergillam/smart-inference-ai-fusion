#!/usr/bin/env python3
"""
Script de teste para validar os novos constraints de verificação formal
(invariantes, pré/pós-condições e robustez) usando o dataset Wine.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from smart_inference_ai_fusion.verification.core.formal_verification import verify
from configs.wine_advanced_verification import (
    get_wine_verification_config,
    get_wine_verification_summary
)

def test_wine_advanced_verification():
    """Testa as novas verificações formais com dados do Wine."""
    
    print("🧪 TESTANDO VERIFICAÇÃO FORMAL AVANÇADA - DATASET WINE")
    print("=" * 70)
    
    # Dados sintéticos que simulam o dataset Wine
    wine_data = np.random.rand(36, 13)  # 36 amostras, 13 features
    wine_labels = np.random.randint(0, 3, 36)  # 3 classes
    wine_probabilities = np.random.rand(36, 3)  # Probabilidades
    wine_probabilities = wine_probabilities / wine_probabilities.sum(axis=1, keepdims=True)
    
    # Parâmetros de exemplo para LogisticRegression
    wine_parameters = {
        'C': 1.0,
        'max_iter': 1000,
        'solver': 'lbfgs',
        'random_state': 42,
        'tol': 1e-6
    }
    
    # Obter configurações de verificação
    summary = get_wine_verification_summary()
    print(f"📊 Resumo das Verificações:")
    print(f"   • Invariantes: {summary['invariants']}")
    print(f"   • Pré-condições: {summary['preconditions']}")
    print(f"   • Pós-condições: {summary['postconditions']}")
    print(f"   • Testes de Robustez: {summary['robustness_tests']}")
    print(f"   • Total de Constraints: {summary['total_constraints']}")
    print(f"   • Algoritmos Suportados: {summary['supported_algorithms']}")
    print()
    
    # Testar cada algoritmo
    algorithms = ['LogisticRegression', 'MLP', 'DecisionTree']
    
    for algorithm in algorithms:
        print(f"🔬 TESTANDO {algorithm.upper()}")
        print("-" * 50)
        
        # Obter configuração específica
        config = get_wine_verification_config(algorithm)
        
        # Teste 1: Invariantes
        print("🔒 1. Testando Invariantes...")
        result_invariants = verify(
            name=f"wine_{algorithm}_invariants",
            constraints={'invariant': config['invariant']},
            input_data=wine_data,
            parameters=wine_parameters,
            timeout=30.0
        )
        print(f"   Status: {result_invariants.status}")
        print(f"   Tempo: {result_invariants.execution_time:.3f}s")
        if result_invariants.message:
            print(f"   Mensagem: {result_invariants.message}")
        print()
        
        # Teste 2: Pré-condições
        print("🔧 2. Testando Pré-condições...")
        result_preconditions = verify(
            name=f"wine_{algorithm}_preconditions",
            constraints={'precondition': config['precondition']},
            input_data=wine_data,
            parameters=wine_parameters,
            timeout=30.0
        )
        print(f"   Status: {result_preconditions.status}")
        print(f"   Tempo: {result_preconditions.execution_time:.3f}s")
        if result_preconditions.message:
            print(f"   Mensagem: {result_preconditions.message}")
        print()
        
        # Teste 3: Pós-condições
        print("⚡ 3. Testando Pós-condições...")
        result_postconditions = verify(
            name=f"wine_{algorithm}_postconditions",
            constraints={'postcondition': config['postcondition']},
            input_data=wine_data,
            output_data=wine_probabilities,
            parameters=wine_parameters,
            timeout=30.0
        )
        print(f"   Status: {result_postconditions.status}")
        print(f"   Tempo: {result_postconditions.execution_time:.3f}s")
        if result_postconditions.message:
            print(f"   Mensagem: {result_postconditions.message}")
        print()
        
        # Teste 4: Robustez
        print("🛡️  4. Testando Robustez...")
        result_robustness = verify(
            name=f"wine_{algorithm}_robustness",
            constraints={'robustness': config['robustness']},
            input_data=wine_data,
            output_data=wine_probabilities,
            parameters=wine_parameters,
            timeout=60.0  # Mais tempo para testes de robustez
        )
        print(f"   Status: {result_robustness.status}")
        print(f"   Tempo: {result_robustness.execution_time:.3f}s")
        if result_robustness.message:
            print(f"   Mensagem: {result_robustness.message}")
        print()
        
        # Teste 5: Verificação Completa
        print("🎯 5. Testando Verificação Completa...")
        result_complete = verify(
            name=f"wine_{algorithm}_complete",
            constraints=config,
            input_data=wine_data,
            output_data=wine_probabilities,
            parameters=wine_parameters,
            timeout=120.0  # Mais tempo para verificação completa
        )
        print(f"   Status: {result_complete.status}")
        print(f"   Tempo: {result_complete.execution_time:.3f}s")
        if result_complete.message:
            print(f"   Mensagem: {result_complete.message}")
        
        if hasattr(result_complete, 'constraints_satisfied'):
            print(f"   Constraints Satisfeitos: {len(result_complete.constraints_satisfied)}")
        if hasattr(result_complete, 'constraints_violated'):
            print(f"   Constraints Violados: {len(result_complete.constraints_violated)}")
        
        print("=" * 50)
        print()
    
    print("✅ TESTE DE VERIFICAÇÃO FORMAL AVANÇADA CONCLUÍDO!")

def test_constraint_violations():
    """Testa casos onde constraints devem ser violados para validar a detecção."""
    
    print("🚨 TESTANDO DETECÇÃO DE VIOLAÇÕES")
    print("=" * 50)
    
    # Dados inválidos (com NaN)
    invalid_data = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]])
    
    # Probabilidades inválidas (fora de [0,1])
    invalid_probabilities = np.array([[1.5, -0.3, 0.8], [0.2, 0.3, 1.2]])
    
    # Parâmetros inválidos
    invalid_parameters = {
        'C': -1.0,  # Deve ser positivo
        'max_iter': 50,  # Muito baixo
        'solver': 'invalid_solver'
    }
    
    config = get_wine_verification_config('LogisticRegression')
    
    # Teste de dados inválidos
    print("1. Testando dados com NaN...")
    result = verify(
        name="wine_invalid_data_test",
        constraints={'invariant': config['invariant']},
        input_data=invalid_data,
        parameters={'C': 1.0, 'max_iter': 1000, 'solver': 'lbfgs', 'random_state': 42},
        timeout=30.0
    )
    print(f"   Status: {result.status} (esperado: FAILURE)")
    print(f"   Mensagem: {result.message}")
    print()
    
    # Teste de probabilidades inválidas
    print("2. Testando probabilidades inválidas...")
    result = verify(
        name="wine_invalid_probabilities_test",
        constraints={'postcondition': config['postcondition']},
        input_data=np.array([[1.0, 2.0, 3.0]]),
        output_data=invalid_probabilities,
        parameters={'C': 1.0, 'max_iter': 1000, 'solver': 'lbfgs', 'random_state': 42},
        timeout=30.0
    )
    print(f"   Status: {result.status} (esperado: FAILURE)")
    print(f"   Mensagem: {result.message}")
    print()
    
    # Teste de parâmetros inválidos
    print("3. Testando parâmetros inválidos...")
    result = verify(
        name="wine_invalid_parameters_test",
        constraints={'invariant': config['invariant']},
        input_data=np.array([[1.0, 2.0, 3.0]]),
        parameters=invalid_parameters,
        timeout=30.0
    )
    print(f"   Status: {result.status} (esperado: FAILURE)")
    print(f"   Mensagem: {result.message}")
    print()

if __name__ == "__main__":
    try:
        # Teste principal
        test_wine_advanced_verification()
        
        # Teste de violações
        test_constraint_violations()
        
    except Exception as e:
        print(f"❌ ERRO durante o teste: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("🎉 TODOS OS TESTES CONCLUÍDOS COM SUCESSO!")