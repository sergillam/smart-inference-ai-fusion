#!/usr/bin/env python3
"""Exemplo de uso da verificação formal integrada nos experimentos."""

import logging
from smart_inference_ai_fusion.experiments.experiment_registry import DIGITS_EXPERIMENTS
from smart_inference_ai_fusion.experiments.common import run_standard_experiment
from smart_inference_ai_fusion.models.random_forest_classifier_model import RandomForestClassifierModel
from smart_inference_ai_fusion.utils.types import (
    DatasetSourceType, 
    SklearnDatasetName, 
    VerificationConfig
)
from smart_inference_ai_fusion.verification.core.formal_verification import (
    enable_verification, 
    disable_verification,
    list_verifiers
)

# Configurar logging
logging.basicConfig(level=logging.INFO)

def exemplo_verificacao_basica():
    """Exemplo básico de uso da verificação formal."""
    print("=== EXEMPLO: Verificação Formal Básica ===")
    
    # Configurar verificação formal
    verification_config = VerificationConfig(
        enabled=True,
        timeout=30.0,
        fail_on_error=False,  # Não falhar em erros de verificação
        constraints={
            'preserve_shape': True,
            'preserve_bounds': True,
            'parameter_validity': True
        }
    )
    
    # Executar experimento com verificação
    baseline_metrics, inference_metrics = run_standard_experiment(
        model_class=RandomForestClassifierModel,
        model_name="RandomForest_with_verification",
        dataset_source=DatasetSourceType.SKLEARN,
        dataset_name=SklearnDatasetName.DIGITS,
        model_params={"n_estimators": 10, "random_state": 42},
        verification_config=verification_config
    )
    
    print("Experimento concluído com verificação formal!")
    print(f"Baseline accuracy: {baseline_metrics.get('accuracy', 'N/A')}")
    print(f"Inference accuracy: {inference_metrics.get('accuracy', 'N/A')}")


def exemplo_verificacao_rigorosa():
    """Exemplo com verificação rigorosa que falha em erros."""
    print("\n=== EXEMPLO: Verificação Rigorosa ===")
    
    # Configurar verificação rigorosa
    verification_config = VerificationConfig(
        enabled=True,
        timeout=60.0,
        fail_on_error=True,  # Falhar se houver erros de verificação
        verifier_name="Z3",  # Usar verificador específico
        constraints={
            'type': 'data_integrity',
            'preserve_shape': True,
            'preserve_bounds': True,
            'bounds_tolerance': 0.05,  # Tolerância menor para bounds
            'parameter_validity': True,
            'parameter_bounds_check': True
        }
    )
    
    try:
        baseline_metrics, inference_metrics = run_standard_experiment(
            model_class=RandomForestClassifierModel,
            model_name="RandomForest_strict_verification",
            dataset_source=DatasetSourceType.SKLEARN,
            dataset_name=SklearnDatasetName.IRIS,
            model_params={"n_estimators": 5, "max_depth": 3, "random_state": 42},
            verification_config=verification_config
        )
        
        print("Experimento passou na verificação rigorosa!")
        print(f"Baseline accuracy: {baseline_metrics.get('accuracy', 'N/A')}")
        print(f"Inference accuracy: {inference_metrics.get('accuracy', 'N/A')}")
        
    except RuntimeError as e:
        print(f"Experimento falhou na verificação: {e}")


def exemplo_sem_verificacao():
    """Exemplo sem verificação formal para comparação."""
    print("\n=== EXEMPLO: Sem Verificação (Controle) ===")
    
    # Desabilitar verificação globalmente
    disable_verification()
    
    baseline_metrics, inference_metrics = run_standard_experiment(
        model_class=RandomForestClassifierModel,
        model_name="RandomForest_no_verification",
        dataset_source=DatasetSourceType.SKLEARN,
        dataset_name=SklearnDatasetName.WINE,
        model_params={"n_estimators": 10, "random_state": 42}
    )
    
    print("Experimento concluído sem verificação!")
    print(f"Baseline accuracy: {baseline_metrics.get('accuracy', 'N/A')}")
    print(f"Inference accuracy: {inference_metrics.get('accuracy', 'N/A')}")
    
    # Reabilitar verificação
    enable_verification()


def exemplo_usando_registry():
    """Exemplo usando o registry de experimentos com verificação."""
    print("\n=== EXEMPLO: Usando Registry com Verificação ===")
    
    # Obter configuração do registry
    config = DIGITS_EXPERIMENTS[RandomForestClassifierModel]
    
    # Adicionar verificação à configuração
    config.verification_config = VerificationConfig(
        enabled=True,
        timeout=30.0,
        fail_on_error=False,
        constraints={
            'preserve_shape': True,
            'parameter_validity': True
        }
    )
    
    # Executar experimento
    baseline_metrics, inference_metrics = run_standard_experiment(
        model_class=config.model_class,
        model_name=config.model_name,
        dataset_source=config.dataset_source,
        dataset_name=config.dataset_name,
        model_params=config.model_params,
        verification_config=config.verification_config
    )
    
    print("Experimento do registry concluído com verificação!")
    print(f"Baseline accuracy: {baseline_metrics.get('accuracy', 'N/A')}")
    print(f"Inference accuracy: {inference_metrics.get('accuracy', 'N/A')}")


def mostrar_verificadores_disponiveis():
    """Mostra os verificadores disponíveis no sistema."""
    print("\n=== VERIFICADORES DISPONÍVEIS ===")
    
    verificadores = list_verifiers()
    if verificadores:
        for nome, info in verificadores.items():
            print(f"- {nome}: {info}")
    else:
        print("Nenhum verificador disponível.")


if __name__ == "__main__":
    print("🔍 SMART INFERENCE AI FUSION - VERIFICAÇÃO FORMAL INTEGRADA")
    print("=" * 70)
    
    # Mostrar verificadores disponíveis
    mostrar_verificadores_disponiveis()
    
    # Executar exemplos
    exemplo_verificacao_basica()
    exemplo_verificacao_rigorosa()
    exemplo_sem_verificacao()
    exemplo_usando_registry()
    
    print("\n✅ Todos os exemplos executados!")
    print("A verificação formal agora está integrada ao sistema de experimentos!")
