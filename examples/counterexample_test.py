"""Exemplo para testar contra-exemplos/contra-provas do Z3."""

import numpy as np
import sys
import os
from dataclasses import asdict

# Adicionar o caminho do projeto ao PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from smart_inference_ai_fusion.verification.core.formal_verification import verification_manager
from smart_inference_ai_fusion.utils.report import report_data, ReportMode


def test_counterexamples():
    """Testa geração de contra-exemplos para constraints violados."""
    print("🚀 Testando Contra-Exemplos no Z3")
    print("=" * 50)
    
    # Dados de teste
    input_data = np.array([5.0, 15.0, -2.0])  # Alguns valores que vão violar constraints
    output_data = np.array([1.0, 2.0, 3.0])
    parameters = {"learning_rate": 0.1, "batch_size": 32}
    
    # 1. Teste com bounds violados
    print("\n🔍 Teste 1: Bounds Violados...")
    bounds_constraints = {
        "bounds": {
            "min": 0.0,
            "max": 10.0,
            "strict": False,
            "allow_nan": False
        }
    }
    
    result = verification_manager.verify(
        name="bounds_violation_test",
        constraints=bounds_constraints,
        input_data=input_data,
        output_data=output_data,
        parameters=parameters
    )
    
    print(f"Status: {result.status}")
    print(f"Violações: {result.constraints_violated}")
    if result.details and "bounds" in result.details:
        counterexample = result.details["bounds"].get("counterexample")
        if counterexample:
            print("🔍 Contra-exemplo encontrado:")
            print(f"   Tipo: {counterexample.get('constraint_type')}")
            if "violation_examples" in counterexample:
                for example in counterexample["violation_examples"]:
                    print(f"   Violação: {example.get('explanation', 'N/A')}")
    
    # 2. Teste com range_check violado
    print("\n🔍 Teste 2: Range Check Violado...")
    range_constraints = {
        "range_check": {
            "type": "discrete",
            "discrete_values": [1, 2, 3, 4],
            "tolerance": 1e-9
        }
    }
    
    result = verification_manager.verify(
        name="range_violation_test",
        constraints=range_constraints,
        input_data=input_data,
        output_data=output_data,
        parameters=parameters
    )
    
    print(f"Status: {result.status}")
    print(f"Violações: {result.constraints_violated}")
    if result.details and "range_check" in result.details:
        counterexample = result.details["range_check"].get("counterexample")
        if counterexample:
            print("🔍 Contra-exemplo encontrado:")
            print(f"   Tipo: {counterexample.get('constraint_type')}")
            if "violation_examples" in counterexample:
                for example in counterexample["violation_examples"]:
                    print(f"   Violação: {example.get('explanation', 'N/A')}")
    
    # 3. Teste com non_negative violado
    print("\n🔍 Teste 3: Non-Negative Violado...")
    non_negative_constraints = {
        "non_negative": True
    }
    
    result = verification_manager.verify(
        name="non_negative_violation_test",
        constraints=non_negative_constraints,
        input_data=input_data,
        output_data=output_data,
        parameters=parameters
    )
    
    print(f"Status: {result.status}")
    print(f"Violações: {result.constraints_violated}")
    if result.details and "non_negative" in result.details:
        counterexample = result.details["non_negative"].get("counterexample")
        if counterexample:
            print("🔍 Contra-exemplo encontrado:")
            print(f"   Tipo: {counterexample.get('constraint_type')}")
            if "violation_examples" in counterexample:
                for example in counterexample["violation_examples"]:
                    print(f"   Violação: {example.get('explanation', 'N/A')}")
    
    # 4. Teste com linear_arithmetic violado
    print("\n🔍 Teste 4: Linear Arithmetic Violado...")
    linear_constraints = {
        "linear_arithmetic": {
            "coefficients": [1, -1],
            "constant": 0
        }
    }
    
    result = verification_manager.verify(
        name="linear_violation_test",
        constraints=linear_constraints,
        input_data=input_data,
        output_data=output_data,
        parameters=parameters
    )
    
    print(f"Status: {result.status}")
    print(f"Violações: {result.constraints_violated}")
    if result.details and "linear_arithmetic" in result.details:
        counterexample = result.details["linear_arithmetic"].get("counterexample")
        if counterexample:
            print("🔍 Contra-exemplo encontrado:")
            print(f"   Tipo: {counterexample.get('constraint_type')}")
            if "violation_examples" in counterexample:
                for example in counterexample["violation_examples"]:
                    print(f"   Violação: {example.get('explanation', 'N/A')}")
    
    print("\n✅ Teste de contra-exemplos concluído!")


if __name__ == "__main__":
    try:
        test_counterexamples()
    except Exception as e:
        print(f"❌ Erro durante o teste: {e}")
        import traceback
        traceback.print_exc()