"""Exemplo prático de uso do sistema multi-solver com dataset Wine."""

import sys
import time
from pathlib import Path

# Adicionar path do projeto
sys.path.insert(0, str(Path(__file__).parent.parent))

from smart_inference_ai_fusion.utils import (
    get_verification_config,
    VerificationMode,
    SolverChoice,
    print_comparison_summary,
    generate_comparison_report
)

from smart_inference_ai_fusion.verification.core.plugin_interface import registry
from smart_inference_ai_fusion.utils.solver_comparison import global_comparison

from examples.wine_advanced_config import WineAdvancedConfig, setup_wine_experiment_environment


def demonstrate_solver_capabilities():
    """Demonstra capacidades dos solvers disponíveis."""
    print("="*60)
    print("🔍 DEMONSTRAÇÃO DOS SOLVERS DISPONÍVEIS")
    print("="*60)
    
    # Verificar solvers disponíveis
    available_verifiers = registry.get_available_verifiers()
    
    print(f"Total de verificadores disponíveis: {len(available_verifiers)}")
    print()
    
    for verifier in available_verifiers:
        print(f"📋 {verifier.name} v{verifier.version}")
        print(f"   Prioridade: {verifier.priority}")
        print(f"   Disponível: {'✅' if verifier.is_available() else '❌'}")
        print(f"   Habilitado: {'✅' if verifier.enabled else '❌'}")
        print(f"   Constraints suportados: {len(verifier.supported_constraints())}")
        
        # Mostrar alguns constraints como exemplo
        constraints = verifier.supported_constraints()
        if constraints:
            print(f"   Exemplos: {', '.join(constraints[:5])}")
            if len(constraints) > 5:
                print(f"   ... e mais {len(constraints) - 5}")
        print()


def run_configuration_examples():
    """Executa exemplos de diferentes configurações."""
    print("="*60)
    print("🎛️ TESTANDO DIFERENTES CONFIGURAÇÕES")
    print("="*60)
    
    configurations = [
        ("Básica", WineAdvancedConfig.BASIC_VERIFICATION),
        ("Comparação", WineAdvancedConfig.COMPARISON_MODE),
        ("Pipeline Completo", WineAdvancedConfig.COMPLETE_PIPELINE),
        ("Performance", WineAdvancedConfig.PERFORMANCE_FOCUSED),
        ("Pesquisa", WineAdvancedConfig.RESEARCH_MODE)
    ]
    
    for config_name, config in configurations:
        print(f"\n🔧 Configuração: {config_name}")
        print(f"   Modo: {config.mode.value}")
        print(f"   Solver: {config.solver.value}")
        print(f"   Timeout por constraint: {config.timeout_per_constraint}s")
        print(f"   Timeout total: {config.timeout_total}s")
        print(f"   Paralelo: {'✅' if config.parallel_solvers else '❌'}")
        print(f"   Comparar solvers: {'✅' if config.compare_solvers else '❌'}")
        print(f"   Log detalhado: {'✅' if config.detailed_logging else '❌'}")
        
        # Verificar quais solvers seriam usados
        enabled_solvers = config.get_enabled_solvers()
        print(f"   Solvers habilitados: {enabled_solvers}")


def simulate_verification_workflow():
    """Simula um workflow de verificação com comparação."""
    print("="*60)
    print("🧪 SIMULANDO WORKFLOW DE VERIFICAÇÃO")
    print("="*60)
    
    # Aplicar configuração de comparação
    WineAdvancedConfig.apply_comparison_mode()
    
    # Simular alguns resultados de verificação
    print("\n📊 Simulando verificações...")
    
    # Simulação de resultados diferentes para demonstração
    mock_results = [
        {
            "experiment": "wine_param_perturbation_1",
            "results": {
                "Z3": {
                    "status": "SUCCESS",
                    "execution_time": 0.025,
                    "constraints_checked": ["type_safety", "bounds"],
                    "constraints_satisfied": ["type_safety", "bounds"],
                    "constraints_violated": []
                },
                "CVC5": {
                    "status": "SUCCESS", 
                    "execution_time": 0.031,
                    "constraints_checked": ["type_safety", "bounds"],
                    "constraints_satisfied": ["type_safety", "bounds"],
                    "constraints_violated": []
                }
            }
        },
        {
            "experiment": "wine_param_perturbation_2",
            "results": {
                "Z3": {
                    "status": "SUCCESS",
                    "execution_time": 0.018,
                    "constraints_checked": ["type_safety", "range_check"],
                    "constraints_satisfied": ["type_safety", "range_check"],
                    "constraints_violated": []
                },
                "CVC5": {
                    "status": "FAILURE",
                    "execution_time": 0.045,
                    "constraints_checked": ["type_safety", "range_check"],
                    "constraints_satisfied": ["type_safety"],
                    "constraints_violated": ["range_check"]
                }
            }
        },
        {
            "experiment": "wine_data_integrity",
            "results": {
                "Z3": {
                    "status": "SUCCESS",
                    "execution_time": 0.033,
                    "constraints_checked": ["bounds", "real_arithmetic"],
                    "constraints_satisfied": ["bounds", "real_arithmetic"],
                    "constraints_violated": []
                },
                "CVC5": {
                    "status": "SUCCESS",
                    "execution_time": 0.029,
                    "constraints_checked": ["bounds", "real_arithmetic"],
                    "constraints_satisfied": ["bounds", "real_arithmetic"],
                    "constraints_violated": []
                }
            }
        }
    ]
    
    # Adicionar resultados simulados ao framework de comparação
    for mock_result in mock_results:
        global_comparison.add_verification_results(
            mock_result["experiment"], 
            mock_result["results"]
        )
        print(f"   ✅ Adicionado: {mock_result['experiment']}")
    
    print(f"\n📈 Total de comparações adicionadas: {len(mock_results)}")


def generate_and_display_report():
    """Gera e exibe relatório de comparação."""
    print("="*60)
    print("📊 GERANDO RELATÓRIO DE COMPARAÇÃO")
    print("="*60)
    
    # Exibir resumo atual
    print("\n📋 Resumo atual:")
    print_comparison_summary()
    
    # Gerar relatório completo
    print("\n📝 Gerando relatório completo...")
    report = generate_comparison_report("wine_demo_experiment")
    
    print(f"\n✅ Relatório gerado: {report.experiment_name}")
    print(f"   Data/Hora: {report.timestamp}")
    print(f"   Solvers: {', '.join(report.solvers_compared)}")
    print(f"   Comparações: {report.total_comparisons}")
    print(f"   Taxa de concordância: {report.agreement_rate:.1%}")
    print(f"   Vencedor performance: {report.performance_winner}")
    print(f"   Vencedor confiabilidade: {report.reliability_winner}")
    
    # Mostrar detalhes dos rankings
    if "solver_rankings" in report.summary:
        rankings = report.summary["solver_rankings"]
        print(f"\n🏆 Rankings:")
        
        if "success_rate" in rankings:
            print(f"   Taxa de Sucesso: {' > '.join(rankings['success_rate'])}")
        
        if "average_speed" in rankings:
            print(f"   Velocidade Média: {' > '.join(rankings['average_speed'])}")
        
        if "reliability" in rankings:
            print(f"   Confiabilidade: {' > '.join(rankings['reliability'])}")


def show_makefile_usage():
    """Mostra exemplos de uso do Makefile."""
    print("="*60)
    print("🔧 EXEMPLOS DE USO COM MAKEFILE")
    print("="*60)
    
    examples = [
        ("Experimento básico", "make run-basic EXP=wine"),
        ("Verificação com Z3", "make run-z3 EXP=wine"),
        ("Verificação com CVC5", "make run-cvc5 EXP=wine"), 
        ("Comparação entre solvers", "make run-both-solvers EXP=wine"),
        ("Pipeline completo", "make run-all EXP=wine"),
        ("Configuração avançada", "make run-advanced EXP=wine"),
        ("Modo debug", "make debug EXP=wine")
    ]
    
    for description, command in examples:
        print(f"\n📋 {description}:")
        print(f"   {command}")
    
    print(f"\n💡 Dicas:")
    print(f"   - Use 'make help' para ver todos os comandos disponíveis")
    print(f"   - Adicione ARGS='--verbose' para logs detalhados")
    print(f"   - Resultados são salvos em results/wine_experiments/")
    print(f"   - Comparações são salvas em results/solver_comparison/")


def main():
    """Função principal do exemplo."""
    print("🍷 SMART INFERENCE AI FUSION - EXEMPLO WINE MULTI-SOLVER")
    print("🔬 Demonstração do sistema de verificação formal multi-solver")
    print()
    
    # Setup do ambiente
    setup_wine_experiment_environment()
    print()
    
    # Demonstrações
    demonstrate_solver_capabilities()
    run_configuration_examples() 
    simulate_verification_workflow()
    generate_and_display_report()
    show_makefile_usage()
    
    print("\n" + "="*60)
    print("✅ DEMONSTRAÇÃO CONCLUÍDA!")
    print("🚀 Agora você pode usar os comandos make para executar experimentos reais.")
    print("📖 Consulte examples/wine_advanced_config.py para mais configurações.")
    print("="*60)


if __name__ == "__main__":
    main()