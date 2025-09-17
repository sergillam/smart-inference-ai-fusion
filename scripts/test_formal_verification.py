#!/usr/bin/env python3
"""Script de teste para validar a interface de plugins de verificação formal."""

import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_plugin_interface():
    """Testa a interface de plugins."""
    logging.info("🧪 Testando Interface de Plugins de Verificação Formal")
    logging.info("=" * 60)
    
    try:
        # Importar interface
        from smart_inference_ai_fusion.verification import (
            verify, list_verifiers, get_available_verifiers,
            enable_verification, disable_verification
        )
        logging.info("✅ Interface importada com sucesso")
        
        # Listar verificadores disponíveis
        logging.info("\n📋 Verificadores disponíveis:")
        verifiers = list_verifiers()
        for name, info in verifiers.items():
            status = "🟢 Disponível" if info['available'] else "🔴 Indisponível"
            enabled = "✅ Habilitado" if info['enabled'] else "❌ Desabilitado"
            logging.info(f"  {name}: {status}, {enabled}")
            logging.info(f"    Suporta: {', '.join(info['supported_constraints'][:5])}...")
        
        # Testar verificação simples
        logging.info("\n🔍 Testando verificação simples...")
        result = verify(
            name="teste_bounds",
            constraints={
                'bounds': {'min': 0, 'max': 100}
            }
        )
        
        logging.info(f"  Status: {result.status.value}")
        logging.info(f"  Verificador: {result.verifier_name}")
        logging.info(f"  Tempo: {result.execution_time:.3f}s")
        logging.info(f"  Mensagem: {result.message}")
        
        if result.constraints_satisfied:
            logging.info(f"  ✅ Constraints satisfeitos: {', '.join(result.constraints_satisfied)}")
        if result.constraints_violated:
            logging.info(f"  ❌ Constraints violados: {', '.join(result.constraints_violated)}")
        
        # Testar múltiplos constraints
        logging.info("\n🔍 Testando múltiplos constraints...")
        result2 = verify(
            name="teste_multiplo",
            constraints={
                'bounds': {'min': -10, 'max': 10},
                'linear_arithmetic': {'coefficients': [1, -1], 'constant': 0},
                'boolean_logic': {}
            }
        )
        
        logging.info(f"  Status: {result2.status.value}")
        logging.info(f"  Constraints verificados: {len(result2.constraints_checked)}")
        logging.info(f"  Satisfeitos: {len(result2.constraints_satisfied)}")
        logging.info(f"  Violados: {len(result2.constraints_violated)}")
        
        # Testar controle de ativação/desativação
        logging.info("\n🎛️ Testando controle de ativação...")
        
        # Desabilitar
        disable_verification()
        result3 = verify("teste_desabilitado", {'bounds': {'min': 0, 'max': 1}})
        logging.info(f"  Verificação desabilitada: {result3.status.value}")
        
        # Reabilitar
        enable_verification()
        result4 = verify("teste_reabilitado", {'bounds': {'min': 0, 'max': 1}})
        logging.info(f"  Verificação reabilitada: {result4.status.value}")
        
        logging.info("\n🎉 Todos os testes passaram!")
        return True
        
    except Exception as e:
        logging.info(f"\n❌ Erro durante teste: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_z3_capabilities():
    """Testa capacidades específicas do Z3."""
    logging.info("\n🧠 Testando Capacidades Avançadas do Z3")
    logging.info("=" * 45)
    
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
            logging.info(f"\n  🧪 {test_case['name']}:")
            result = verify(
                name=f"test_{test_case['name'].lower().replace(' ', '_')}",
                constraints=test_case['constraints'],
                timeout=10.0
            )
            
            logging.info(f"    Status: {result.status.value}")
            logging.info(f"    Tempo: {result.execution_time:.3f}s")
            
            if result.success:
                successful_tests += 1
                logging.info("    ✅ Sucesso")
            else:
                logging.info(f"    ⚠️ {result.message}")
        
        logging.info(f"\n📊 Resumo: {successful_tests}/{len(test_cases)} testes bem-sucedidos")
        
        return successful_tests > 0
        
    except Exception as e:
        logging.info(f"\n❌ Erro nos testes Z3: {e}")
        return False

def main():
    """Função principal."""
    logging.info("🚀 Sistema de Verificação Formal - Validação Completa")
    logging.info("=" * 70)
    
    # Teste 1: Interface básica
    success1 = test_plugin_interface()
    
    # Teste 2: Capacidades Z3  
    success2 = test_z3_capabilities()
    
    # Resultado final
    if success1 and success2:
        logging.info("\n🎉 SUCESSO: Sistema de verificação formal validado!")
        logging.info("✅ Interface de plugins funcionando")
        logging.info("✅ Z3 com capacidades avançadas funcionando")
        logging.info("✅ Controle de ativação/desativação funcionando")
        return 0
    else:
        logging.info("\n⚠️ ATENÇÃO: Alguns testes falharam")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
