"""Exemplo demonstrando constrai    # Test 1: Valid bounds
    result = verification_manager.verify(
        name="bounds_valid_test",
        constraints=bounds_constraints,
        input_data=input_data,
        output_data=output_data,
        parameters=parameters
    )
    
    report_data(asdict(result), ReportMode.PRINT, "Bounds Valid Test")turados para verificação formal."""

import numpy as np
import sys
import os
from dataclasses import asdict

# Adicionar o caminho do projeto ao PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from smart_inference_ai_fusion.verification.core.formal_verification import verification_manager
from smart_inference_ai_fusion.verification.core.plugin_interface import VerificationInput
from smart_inference_ai_fusion.utils.report import report_data, ReportMode

def test_bounds_constraints():
    """Testa constraints de bounds estruturados."""
    print("🔍 Testando Bounds Constraints Estruturados...")
    
    # Dados de teste
    valid_data = np.array([0.5, 1.0, 2.5, 5.0])
    invalid_data = np.array([0.5, 15.0, -5.0, 100.0])  # Alguns valores fora dos bounds
    
    # Constraint estruturado
    bounds_constraint = {
        'bounds': {
            'min': 0.0,
            'max': 10.0,
            'strict': False,
            'allow_nan': False
        }
    }
    
    # Teste 1: Dados válidos
    result = verification_manager.verify(
        name="bounds_valid_test",
        constraints=bounds_constraint,
        input_data=None,
        output_data=valid_data,
        parameters={'test_type': 'valid_bounds'}
    )
    report_data(asdict(result), ReportMode.PRINT, "Bounds Valid Test")
    
    # Teste 2: Dados inválidos
    result = verification_manager.verify(
        name="bounds_invalid_test",
        constraints=bounds_constraint,
        input_data=None,
        output_data=invalid_data,
        parameters={'test_type': 'invalid_bounds'}
    )
    report_data(asdict(result), ReportMode.PRINT, "Bounds Invalid Test")

def test_range_check_constraints():
    """Testa constraints de range check estruturados."""
    print("🔍 Testando Range Check Constraints Estruturados...")
    
    # Teste para range contínuo
    continuous_data = np.array([0.1, 0.5, 0.9, 1.1, 2.0])
    
    range_constraint_continuous = {
        'range_check': {
            'type': 'continuous',
            'valid_ranges': [(0.0, 1.0), (1.5, 2.5)],
            'tolerance': 1e-6
        }
    }
    
    result = verification_manager.verify(
        name="range_continuous_test",
        constraints=range_constraint_continuous,
        input_data=None,
        output_data=continuous_data,
        parameters={'test_type': 'continuous_range'}
    )
    report_data(asdict(result), ReportMode.PRINT, "Range Continuous Test")
    
    # Teste para range discreto
    discrete_data = np.array([0, 1, 2, 5, 10])
    
    range_constraint_discrete = {
        'range_check': {
            'type': 'discrete',
            'discrete_values': [0, 1, 2, 3, 4, 5],
            'tolerance': 1e-9
        }
    }
    
    result = verification_manager.verify(
        name="range_discrete_test",
        constraints=range_constraint_discrete,
        input_data=None,
        output_data=discrete_data,
        parameters={'test_type': 'discrete_range'}
    )
    report_data(asdict(result), ReportMode.PRINT, "Range Discrete Test")

def test_complex_constraints():
    """Testa constraints complexos combinados."""
    print("🔍 Testando Constraints Complexos Combinados...")
    
    # Dados de teste
    test_data = np.array([0.1, 0.5, 0.8, 1.2, 2.0, 15.0])
    
    # Constraints múltiplos
    complex_constraints = {
        'bounds': {
            'min': 0.0,
            'max': 10.0,
            'strict': False,
            'allow_nan': False
        },
        'range_check': {
            'type': 'continuous',
            'valid_ranges': [(0.0, 1.0), (1.0, 3.0)],
            'tolerance': 1e-6
        },
        'shape_preservation': True,
        'type_safety': True
    }
    
    result = verification_manager.verify(
        name="complex_constraints_test",
        constraints=complex_constraints,
        input_data=None,
        output_data=test_data,
        parameters={'test_type': 'complex_multiple'}
    )
    report_data(asdict(result), ReportMode.PRINT, "Complex Constraints Test")

def test_nan_and_special_values():
    """Testa tratamento de NaN e valores especiais."""
    print("🔍 Testando NaN e Valores Especiais...")
    
    # Dados com NaN
    nan_data = np.array([0.5, np.nan, 2.0, np.inf, -np.inf])
    
    # Constraint que permite NaN
    allow_nan_constraint = {
        'bounds': {
            'min': 0.0,
            'max': 10.0,
            'strict': False,
            'allow_nan': True
        }
    }
    
    result = verification_manager.verify(
        name="nan_allowed_test",
        constraints=allow_nan_constraint,
        input_data=None,
        output_data=nan_data,
        parameters={'test_type': 'nan_allowed'}
    )
    report_data(asdict(result), ReportMode.PRINT, "NaN Allowed Test")
    
    # Constraint que não permite NaN
    no_nan_constraint = {
        'bounds': {
            'min': 0.0,
            'max': 10.0,
            'strict': False,
            'allow_nan': False
        }
    }
    
    result = verification_manager.verify(
        name="nan_disallowed_test",
        constraints=no_nan_constraint,
        input_data=None,
        output_data=nan_data,
        parameters={'test_type': 'nan_disallowed'}
    )
    report_data(asdict(result), ReportMode.PRINT, "NaN Disallowed Test")

if __name__ == "__main__":
    # Habilitar verificação
    verification_manager.enable_verification()
    
    print("🚀 Testando Constraints Estruturados para Verificação Formal\n")
    
    try:
        test_bounds_constraints()
        print("\n" + "="*60 + "\n")
        
        test_range_check_constraints()
        print("\n" + "="*60 + "\n")
        
        test_complex_constraints()
        print("\n" + "="*60 + "\n")
        
        test_nan_and_special_values()
        
        print("\n✅ Todos os testes de constraints estruturados concluídos!")
        
    except Exception as e:
        print(f"❌ Erro durante os testes: {e}")
        import traceback
        traceback.print_exc()