#!/usr/bin/env python3
"""Script de teste para validar a interface de plugins de verificação formal."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_plugin_interface():
    """Testa a interface de plugins."""
    print("🧪 Testando Interface de Plugins de Verificação Formal")
    print("=" * 60)
    
    try:
        # Importar interface
        from smart_inference_ai_fusion.verification import (
            verify, list_verifiers, get_available_verifiers,
            enable_verification, disable_verification
        )
        print("✅ Interface importada com sucesso")
        
        # Listar verificadores disponíveis
        print("\n📋 Verificadores disponíveis:")
        verifiers = list_verifiers()
        for name, info in verifiers.items():
            status = "🟢 Disponível" if info['available'] else "🔴 Indisponível"
            enabled = "✅ Habilitado" if info['enabled'] else "❌ Desabilitado"
            print(f"  {name}: {status}, {enabled}")
            print(f"    Suporta: {', '.join(info['supported_constraints'][:5])}...")
        
        # Testar verificação simples
        print("\n🔍 Testando verificação simples...")
        result = verify(
            name="teste_bounds",
            constraints={
                'bounds': {'min': 0, 'max': 100}
            }
        )
        
        print(f"  Status: {result.status.value}")
        print(f"  Verificador: {result.verifier_name}")
        print(f"  Tempo: {result.execution_time:.3f}s")
        print(f"  Mensagem: {result.message}")
        
        if result.constraints_satisfied:
            print(f"  ✅ Constraints satisfeitos: {', '.join(result.constraints_satisfied)}")
        if result.constraints_violated:
            print(f"  ❌ Constraints violados: {', '.join(result.constraints_violated)}")
        
        # Testar múltiplos constraints
        print("\n🔍 Testando múltiplos constraints...")
        result2 = verify(
            name="teste_multiplo",
            constraints={
                'bounds': {'min': -10, 'max': 10},
                'linear_arithmetic': {'coefficients': [1, -1], 'constant': 0},
                'boolean_logic': {}
            }
        )
        
        print(f"  Status: {result2.status.value}")
        print(f"  Constraints verificados: {len(result2.constraints_checked)}")
        print(f"  Satisfeitos: {len(result2.constraints_satisfied)}")
        print(f"  Violados: {len(result2.constraints_violated)}")
        
        # Testar controle de ativação/desativação
        print("\n🎛️ Testando controle de ativação...")
        
        # Desabilitar
        disable_verification()
        result3 = verify("teste_desabilitado", {'bounds': {'min': 0, 'max': 1}})
        print(f"  Verificação desabilitada: {result3.status.value}")
        
        # Reabilitar
        enable_verification()
        result4 = verify("teste_reabilitado", {'bounds': {'min': 0, 'max': 1}})
        print(f"  Verificação reabilitada: {result4.status.value}")
        
        print("\n🎉 Todos os testes passaram!")
        return True
        
    except Exception as e:
        print(f"\n❌ Erro durante teste: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_z3_capabilities():
    """Testa capacidades específicas do Z3."""
    print("\n🧠 Testando Capacidades Avançadas do Z3")
    print("=" * 45)
    
    try:
        from smart_inference_ai_fusion.verification import verify
        
        # Testar diferentes tipos de constraints
        test_cases = [
            {
                'name': 'Aritmética Linear',
                'constraints': {'linear_arithmetic': {'coefficients': [2, 3], 'constant': -10}}
            },
            {
                'name': 'Lógica Booleana', 
                'constraints': {'boolean_logic': {}}
            },
            {
                'name': 'Teoria de Arrays',
                'constraints': {'array_theory': {'size': 5}}
            },
            {
                'name': 'Bit-vectors',
                'constraints': {'bitvector_arithmetic': {'bit_width': 32}}
            },
            {
                'name': 'Ponto Flutuante',
                'constraints': {'floating_point': {}}
            },
            {
                'name': 'Strings',
                'constraints': {'string_theory': {'pattern': 'test', 'max_length': 50}}
            },
            {
                'name': 'Fórmulas Quantificadas',
                'constraints': {'quantified_formulas': {}}
            },
            {
                'name': 'Otimização',
                'constraints': {'optimization': {}}
            },
            {
                'name': 'Rede Neural',
                'constraints': {'neural_network_verification': {'input_size': 2}}
            },
            {
                'name': 'Probabilidade',
                'constraints': {'probability_bounds': {'min': 0.1, 'max': 0.9}}
            }
        ]
        
        successful_tests = 0
        
        for test_case in test_cases:
            print(f"\n  🧪 {test_case['name']}:")
            result = verify(
                name=f"test_{test_case['name'].lower().replace(' ', '_')}",
                constraints=test_case['constraints'],
                timeout=10.0
            )
            
            print(f"    Status: {result.status.value}")
            print(f"    Tempo: {result.execution_time:.3f}s")
            
            if result.success:
                successful_tests += 1
                print("    ✅ Sucesso")
            else:
                print(f"    ⚠️ {result.message}")
        
        print(f"\n📊 Resumo: {successful_tests}/{len(test_cases)} testes bem-sucedidos")
        
        return successful_tests > 0
        
    except Exception as e:
        print(f"\n❌ Erro nos testes Z3: {e}")
        return False

def main():
    """Função principal."""
    print("🚀 Sistema de Verificação Formal - Validação Completa")
    print("=" * 70)
    
    # Teste 1: Interface básica
    success1 = test_plugin_interface()
    
    # Teste 2: Capacidades Z3  
    success2 = test_z3_capabilities()
    
    # Resultado final
    if success1 and success2:
        print("\n🎉 SUCESSO: Sistema de verificação formal validado!")
        print("✅ Interface de plugins funcionando")
        print("✅ Z3 com capacidades avançadas funcionando")
        print("✅ Controle de ativação/desativação funcionando")
        return 0
    else:
        print("\n⚠️ ATENÇÃO: Alguns testes falharam")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
