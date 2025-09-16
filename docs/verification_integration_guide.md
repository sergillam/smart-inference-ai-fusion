# Integração de Verificação Formal nos Experimentos

Este documento explica como usar a verificação formal integrada no sistema Smart Inference AI Fusion.

## 🎯 **Visão Geral**

A verificação formal agora está completamente integrada ao sistema de experimentos, permitindo:

- **Verificação Automática**: Aplicada em cada etapa do pipeline de inferência
- **Configuração Flexível**: Controle fino sobre constraints e comportamento
- **Múltiplos Verificadores**: Suporte a diferentes engines de verificação (Z3, etc.)
- **Relatórios Detalhados**: Logs de verificação integrados aos resultados

## 🚀 **Como Usar**

### **1. Verificação Básica**

```python
from smart_inference_ai_fusion.experiments.common import run_standard_experiment
from smart_inference_ai_fusion.utils.types import VerificationConfig, DatasetSourceType, SklearnDatasetName
from smart_inference_ai_fusion.models.random_forest_classifier_model import RandomForestClassifierModel

# Configurar verificação
verification_config = VerificationConfig(
    enabled=True,
    timeout=30.0,
    fail_on_error=False,
    constraints={
        'preserve_shape': True,
        'preserve_bounds': True,
        'parameter_validity': True
    }
)

# Executar experimento com verificação
baseline, inference = run_standard_experiment(
    model_class=RandomForestClassifierModel,
    model_name="RandomForest",
    dataset_source=DatasetSourceType.SKLEARN,
    dataset_name=SklearnDatasetName.DIGITS,
    model_params={"n_estimators": 10, "random_state": 42},
    verification_config=verification_config
)
```

### **2. Usando Registry de Experimentos**

```python
from smart_inference_ai_fusion.experiments.experiment_registry import DIGITS_EXPERIMENTS

# Obter configuração existente
config = DIGITS_EXPERIMENTS[RandomForestClassifierModel]

# Adicionar verificação
config.verification_config = VerificationConfig(
    enabled=True,
    timeout=30.0,
    constraints={'preserve_shape': True}
)

# Executar
baseline, inference = run_standard_experiment(
    model_class=config.model_class,
    model_name=config.model_name,
    dataset_source=config.dataset_source,
    dataset_name=config.dataset_name,
    model_params=config.model_params,
    verification_config=config.verification_config
)
```

### **3. Configuração Avançada**

```python
# Verificação rigorosa com verificador específico
verification_config = VerificationConfig(
    enabled=True,
    timeout=60.0,
    fail_on_error=True,  # Falhar se verificação encontrar problemas
    verifier_name="Z3",  # Usar verificador específico
    constraints={
        'type': 'data_integrity',
        'preserve_shape': True,
        'preserve_bounds': True,
        'bounds_tolerance': 0.05,  # Tolerância de 5%
        'parameter_validity': True,
        'parameter_bounds_check': True,
        'parameter_bounds': {
            'n_estimators': {'min': 1, 'max': 1000},
            'max_depth': {'min': 1, 'max': 50}
        }
    }
)
```

## ⚙️ **Opções de Configuração**

### **VerificationConfig**

| Campo | Tipo | Padrão | Descrição |
|-------|------|--------|-----------|
| `enabled` | `bool` | `True` | Habilita/desabilita verificação |
| `timeout` | `float` | `30.0` | Timeout em segundos |
| `fail_on_error` | `bool` | `False` | Falhar quando encontrar erros |
| `verifier_name` | `str` | `None` | Verificador específico (auto-select se None) |
| `constraints` | `dict` | `{}` | Constraints customizados |

### **Constraints Disponíveis**

#### **Dados (Data)**
- `preserve_shape`: Manter dimensões dos dados
- `preserve_bounds`: Manter limites min/max dos valores
- `bounds_tolerance`: Tolerância para mudanças de bounds (padrão: 0.1)

#### **Labels**
- `preserve_class_distribution`: Manter distribuição de classes
- `distribution_tolerance`: Tolerância para mudanças de distribuição (padrão: 0.05)

#### **Parâmetros**
- `parameter_validity`: Verificar validade dos parâmetros
- `preserve_parameter_types`: Manter tipos dos parâmetros
- `parameter_bounds_check`: Verificar bounds dos parâmetros
- `parameter_bounds`: Definir bounds específicos

## 🔍 **Pontos de Verificação**

A verificação é aplicada automaticamente em:

### **1. Pipeline de Dados**
- **Entrada**: Integridade dos dados originais
- **Saída**: Preservação de propriedades após transformações

### **2. Pipeline de Labels**
- **Entrada**: Integridade dos labels originais
- **Saída**: Preservação da distribuição de classes

### **3. Pipeline de Parâmetros**
- **Pré-perturbação**: Validade dos parâmetros base
- **Pós-perturbação**: Validade dos parâmetros perturbados
- **Modelo**: Integridade do modelo criado

## 📊 **Verificadores Específicos**

### **DataIntegrityVerifier**
- Verifica preservação de forma dos dados
- Verifica preservação de bounds/limites
- Detecta mudanças excessivas nos dados

### **LabelIntegrityVerifier**
- Verifica distribuição de classes
- Detecta mudanças inválidas em labels
- Preserva balanceamento de classes

### **ParameterIntegrityVerifier**
- Verifica tipos de parâmetros
- Verifica bounds de parâmetros
- Detecta parâmetros inválidos

### **TransformationVerifier**
- Verifica injeção de outliers
- Verifica transformações específicas
- Detecta anomalias em transformações

## 🎛️ **Controle Global**

```python
from smart_inference_ai_fusion.verification.core.formal_verification import (
    enable_verification, 
    disable_verification,
    list_verifiers
)

# Controle global
enable_verification()   # Habilitar globalmente
disable_verification()  # Desabilitar globalmente

# Ver verificadores disponíveis
verificadores = list_verifiers()
print(verificadores)
```

## 🚨 **Tratamento de Erros**

### **Modo Não-Rigoroso** (`fail_on_error=False`)
- Logs de verificação são gerados
- Experimento continua mesmo com falhas
- Resultados incluem status de verificação

### **Modo Rigoroso** (`fail_on_error=True`)
- Experimento para se verificação falhar
- `RuntimeError` é lançado com detalhes
- Útil para validação crítica

## 📝 **Logs e Relatórios**

A verificação gera logs detalhados:

```
INFO - Running verification 'pipeline.data.input_data' with Z3
INFO - Verification for apply_data_inference: PASSED
DEBUG - Verification message: Shape preserved successfully
```

Os resultados de verificação são incluídos nos metadados dos experimentos.

## 🔧 **Extensibilidade**

### **Adicionando Verificadores Personalizados**
```python
# Implementar na pasta smart_inference_ai_fusion/verification/plugins/
# Seguir interface FormalVerifier
```

### **Constraints Personalizados**
```python
verification_config = VerificationConfig(
    constraints={
        'custom_constraint': 'custom_value',
        'domain_specific_check': True,
        'business_rule_validation': {'rule': 'value'}
    }
)
```

## ✅ **Exemplos Completos**

Veja `examples/verification_integration_example.py` para exemplos completos de uso.

## 🎯 **Benefícios**

1. **Confiabilidade**: Garantia de integridade dos dados e transformações
2. **Debugging**: Detecção precoce de problemas em transformações
3. **Compliance**: Verificação formal para requisitos críticos
4. **Transparência**: Logs detalhados de todas as verificações
5. **Flexibilidade**: Configuração granular por experimento

A integração está completa e pronta para uso em produção! 🚀
