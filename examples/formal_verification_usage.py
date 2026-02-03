"""Exemplo prático de uso do sistema de verificação formal."""

def example_usage():
    """Demonstra como usar o sistema de verificação formal."""
    
    print("🚀 Exemplo de Uso do Sistema de Verificação Formal")
    print("=" * 60)
    
    # Importar o sistema de verificação
    from smart_inference_ai_fusion.verification import (
        verify, list_verifiers, enable_verification, disable_verification
    )
    
    # 1. Listar verificadores disponíveis
    print("\n1. 📋 Verificadores Disponíveis:")
    verifiers = list_verifiers()
    for name, info in verifiers.items():
        print(f"   • {name}: {'✅' if info['available'] else '❌'}")
        print(f"     Suporta {len(info['supported_constraints'])} tipos de constraints")
    
    # 2. Verificação simples - bounds checking
    print("\n2. 🔍 Verificação de Bounds:")
    result = verify(
        name="bounds_example",
        constraints={
            'bounds': {'min': 0, 'max': 100}
        }
    )
    print(f"   Status: {result.status.value}")
    print(f"   Mensagem: {result.message}")
    
    # 3. Verificação de aritmética linear
    print("\n3. 🔢 Verificação de Aritmética Linear:")
    result = verify(
        name="linear_example", 
        constraints={
            'linear_arithmetic': {
                'coefficients': [2, -3],  # 2x - 3y + 5 <= 0
                'constant': 5
            }
        }
    )
    print(f"   Status: {result.status.value}")
    print(f"   Tempo de execução: {result.execution_time:.3f}s")
    
    # 4. Verificação de rede neural
    print("\n4. 🧠 Verificação de Rede Neural:")
    result = verify(
        name="neural_network_example",
        constraints={
            'neural_network_verification': {
                'input_size': 3
            }
        }
    )
    print(f"   Status: {result.status.value}")
    print(f"   Constraints verificados: {len(result.constraints_checked)}")
    
    # 5. Múltiplos constraints
    print("\n5. 🎯 Múltiplos Constraints:")
    result = verify(
        name="multiple_constraints",
        constraints={
            'bounds': {'min': -10, 'max': 10},
            'probability_bounds': {'min': 0.0, 'max': 1.0},
            'boolean_logic': {},
            'string_theory': {'pattern': 'valid', 'max_length': 100}
        },
        timeout=15.0
    )
    print(f"   Status: {result.status.value}")
    print(f"   Satisfeitos: {result.constraints_satisfied}")
    print(f"   Violados: {result.constraints_violated}")
    
    # 6. Controle de ativação
    print("\n6. 🎛️ Controle de Ativação:")
    
    # Desabilitar
    disable_verification()
    result = verify("test_disabled", {'bounds': {'min': 0, 'max': 1}})
    print(f"   Desabilitado: {result.status.value}")
    
    # Reabilitar
    enable_verification()
    result = verify("test_enabled", {'bounds': {'min': 0, 'max': 1}})
    print(f"   Reabilitado: {result.status.value}")
    
    print("\n✅ Exemplo concluído! O sistema está pronto para uso.")

if __name__ == "__main__":
    example_usage()
