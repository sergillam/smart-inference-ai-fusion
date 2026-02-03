# Guia de Integração da Verificação Formal

Estado referente ao branch `solver-interface` (2025-10). Este guia mostra como ativar, configurar e estender a verificação formal multi-solver (Z3, CVC5) integrada aos experimentos e ao pipeline de inferência.

## 🎯 Visão Geral

Objetivos principais:
- Garantir propriedades básicas das transformações (ex: `bounds`, `shape_preservation`)
- Validar coerência de parâmetros (`parameter_validity`)
- Suportar múltiplos solvers com seleção automática ou explícita
- Manter overhead controlado para uso em experimentos repetitivos

## 🚀 **Como Usar**

### 1. Verificação Básica

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
        'shape_preservation': True,
        'bounds': {'min': 0, 'max': 1},
        'parameter_validity': {'required': ['n_estimators']}
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

### 2. Usando Registry de Experimentos

```python
from smart_inference_ai_fusion.experiments.experiment_registry import DIGITS_EXPERIMENTS

# Obter configuração existente
config = DIGITS_EXPERIMENTS[RandomForestClassifierModel]

# Adicionar verificação
config.verification_config = VerificationConfig(
    enabled=True,
    timeout=30.0,
    constraints={'shape_preservation': True}
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

### 3. Configuração Avançada (Solver específico + tolerâncias)

```python
# Verificação rigorosa com verificador específico
verification_config = VerificationConfig(
    enabled=True,
    timeout=60.0,
    fail_on_error=True,  # Falhar se verificação encontrar problemas
    verifier_name="Z3",
    constraints={
        'shape_preservation': True,
        'bounds': {'min': 0, 'max': 1, 'tolerance': 0.05},
        'parameter_validity': {
            'required': ['n_estimators', 'max_depth'],
            'bounds': {
                'n_estimators': {'min': 1, 'max': 1000},
                'max_depth': {'min': 1, 'max': 50}
            }
        }
    }
)
```

## ⚙️ Opções de Configuração

### VerificationConfig

| Campo | Tipo | Padrão | Descrição |
|-------|------|--------|-----------|
| `enabled` | `bool` | `True` | Habilita/desabilita verificação |
| `timeout` | `float` | `30.0` | Timeout em segundos |
| `fail_on_error` | `bool` | `False` | Falhar quando encontrar erros |
| `verifier_name` | `str` | `None` | Verificador específico (auto-select se None) |
| `constraints` | `dict` | `{}` | Dicionário de constraints (usar chaves suportadas por solver) |

### Chaves de Constraints Básicas (Usuais)

| Chave | Escopo | Payload Esperado | Uso Atual |
|-------|--------|------------------|-----------|
| `bounds` | Dados | `{min: float, max: float, (opcional) tolerance: float}` | Compara extremos antes/depois |
| `shape_preservation` | Dados | `True` | Garante shape idêntico |
| `parameter_validity` | Parâmetros | `{required: [...], bounds: {...}}` | Checa presença e limites declarados |

Outras chaves listadas pelos plugins existem como taxonomia para futura expansão.

## 🔍 Pontos de Verificação no Pipeline

A verificação é aplicada automaticamente em:

### 1. Dados
- **Entrada**: Integridade dos dados originais
- **Saída**: Preservação de propriedades após transformações

### 2. Labels
- **Entrada**: Integridade dos labels originais
- **Saída**: Preservação da distribuição de classes

### 3. Parâmetros
- **Pré-perturbação**: Validade dos parâmetros base
- **Pós-perturbação**: Validade dos parâmetros perturbados
- **Modelo**: Integridade do modelo criado

## 📊 Verificadores Disponíveis

Atualmente a arquitetura utiliza diretamente `FormalVerifier` SMT (Z3 / CVC5). Verificadores conceituais (ex: `DataIntegrityVerifier`) podem ser adicionados posteriormente como wrappers que traduzem verificações de alto nível em constraints SMT delimitadas.

## 🎛️ Controle Global

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

## 🚨 Tratamento de Erros

### Modo Não-Rigoroso (`fail_on_error=False`)
- Logs de verificação são gerados
- Experimento continua mesmo com falhas
- Resultados incluem status de verificação

### Modo Rigoroso (`fail_on_error=True`)
- Experimento para se verificação falhar
- `RuntimeError` é lançado com detalhes
- Útil para validação crítica

## 📝 Logs e Relatórios

A verificação gera logs detalhados:

```
INFO - Running verification 'pipeline.data.input_data' with Z3
INFO - Verification for apply_data_inference: PASSED
DEBUG - Verification message: Shape preserved successfully
```

Os resultados de verificação são incluídos nos metadados dos experimentos.

## 🔧 Extensibilidade

### Adicionando Verificadores Personalizados
```python
# Implementar na pasta smart_inference_ai_fusion/verification/plugins/
# Seguir interface FormalVerifier
```

### Constraints Personalizados
```python
verification_config = VerificationConfig(
    constraints={
        'custom_constraint': 'custom_value',
        'domain_specific_check': True,
        'business_rule_validation': {'rule': 'value'}
    }
)
```

## ✅ Exemplos Completos

Veja `examples/verification_integration_example.py` para exemplos completos de uso.

## 🎯 Benefícios

1. **Confiabilidade**: Garantia de integridade dos dados e transformações
2. **Debugging**: Detecção precoce de problemas em transformações
3. **Compliance**: Verificação formal para requisitos críticos
4. **Transparência**: Logs detalhados de todas as verificações
5. **Flexibilidade**: Configuração granular por experimento

A integração está completa e pronta para uso em produção! 🚀

## 🔌 Multi-Solver (Seleção Dinâmica)

O sistema suporta múltiplos verificadores (Z3, CVC5, etc) e permite seleção dinâmica via parâmetro `verifier_name`:

```python
verification_config = VerificationConfig(
    enabled=True,
    timeout=30.0,
    verifier_name="CVC5",
    constraints={
        'shape_preservation': True,
        'bounds': {'min': 0, 'max': 1}
    }
)
```

## 🚀 Registro Dinâmico e Automação

Experimentos podem ser registrados dinamicamente e integrados ao pipeline de automação:

```python
from smart_inference_ai_fusion.experiments import experiment_registry
experiment_registry["meu_dataset"] = "scripts/meu_experimento.py"
```

## 🧩 Extensibilidade via Plugins

Para adicionar novos verificadores ou constraints, basta implementar um plugin seguindo a interface `FormalVerifier` e registrar no sistema.

## 📊 Benchmarks Automatizados (Futuro)

Com a arquitetura atual, é possível automatizar benchmarks entre diferentes verificadores e configurações, facilitando comparações e validação científica.

---

## 📦 (Opcional) Definição de Constraints via YAML

Proposta (futuro) para carregar constraints por arquivo:

```yaml
step: apply_data_inference
constraints:
    bounds: {min: 0, max: 1, tolerance: 0.05}
    shape_preservation: true
    parameter_validity:
        required: [n_estimators, max_depth]
        bounds:
            n_estimators: {min: 1, max: 500}
```

Carregamento (exemplo conceitual):
```python
import yaml
from smart_inference_ai_fusion.verification import verify

with open('constraints.yaml') as f:
        cfg = yaml.safe_load(f)

verify(name=cfg['step'], constraints=cfg['constraints'])
```

---
Guia revisado para refletir nomenclatura de constraints e multi-solver atual.
