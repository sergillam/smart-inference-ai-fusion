"""Exemplo para forçar violações e testar contra-exemplos."""

import numpy as np
import sys
import os

# Adicionar o caminho do projeto ao PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from smart_inference_ai_fusion.verification.core.formal_verification import verification_manager


def test_forced_counterexamples():
    """Força violações para testar contra-exemplos."""
    print("🚀 Testando Contra-Exemplos Forçados")
    print("=" * 50)
    
    # 1. Teste com bounds impossíveis
    print("\n🔍 Teste 1: Bounds Impossíveis...")
    impossible_bounds = {
        "bounds": {
            "min": 100.0,  # Mínimo muito alto
            "max": 0.0,    # Máximo muito baixo (impossível: min > max)
            "strict": True,
            "allow_nan": False
        }
    }
    
    result = verification_manager.verify(
        name="impossible_bounds_test",
        constraints=impossible_bounds,
        input_data=np.array([5.0, 10.0]),
        output_data=np.array([1.0, 2.0]),
        parameters={"test": True}
    )
    
    print(f"Status: {result.status}")
    print(f"Violações: {result.constraints_violated}")
    print(f"Satisfeitos: {result.constraints_satisfied}")
    
    # 2. Teste com range discreto vazio
    print("\n🔍 Teste 2: Range Discreto Vazio...")
    empty_discrete = {
        "range_check": {
            "type": "discrete",
            "discrete_values": [],  # Lista vazia - impossível satisfazer
            "allow_empty": False
        }
    }
    
    result = verification_manager.verify(
        name="empty_discrete_test",
        constraints=empty_discrete,
        input_data=np.array([1.0, 2.0, 3.0]),
        output_data=np.array([1.0, 2.0, 3.0]),
        parameters={"test": True}
    )
    
    print(f"Status: {result.status}")
    print(f"Violações: {result.constraints_violated}")
    print(f"Satisfeitos: {result.constraints_satisfied}")
    
    # 3. Teste forçando violação através de dados específicos
    print("\n🔍 Teste 3: Violação através de Z3 direto...")
    
    # Vamos usar o verificador Z3 diretamente para criar um cenário de violação
    from smart_inference_ai_fusion.verification.plugins.z3_plugin import Z3Verifier
    
    z3_verifier = Z3Verifier()
    
    # Testar geração de contra-exemplo diretamente
    print("\n🔍 Gerando contra-exemplo para bounds...")
    bounds_data = {
        "min": 0.0,
        "max": 10.0,
        "strict": False,
        "allow_nan": False
    }
    
    counterexample = z3_verifier._generate_counterexample("bounds", bounds_data)
    print(f"Contra-exemplo gerado: {counterexample}")
    
    print("\n🔍 Gerando contra-exemplo para range_check...")
    range_data = {
        "type": "discrete",
        "discrete_values": [1, 2, 3],
        "tolerance": 1e-9
    }
    
    counterexample = z3_verifier._generate_counterexample("range_check", range_data)
    print(f"Contra-exemplo gerado: {counterexample}")
    
    print("\n🔍 Gerando contra-exemplo para non_negative...")
    counterexample = z3_verifier._generate_counterexample("non_negative", True)
    print(f"Contra-exemplo gerado: {counterexample}")
    
    print("\n✅ Teste de contra-exemplos forçados concluído!")


if __name__ == "__main__":
    try:
        test_forced_counterexamples()
    except Exception as e:
        print(f"❌ Erro durante o teste: {e}")
        import traceback
        traceback.print_exc()