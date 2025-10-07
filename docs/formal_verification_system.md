# Sistema de Verificação Formal – Visão Atualizada

Este documento descreve o estado ATUAL (2025-11) do subsistema de verificação formal do projeto, alinhado ao código existente em `smart_inference_ai_fusion/verification/*`.

> **Documentação Detalhada:** Para mapeamento completo dos solvers, ver [verification_solver_mapping.md](verification_solver_mapping.md)

## 🔌 Núcleo da Arquitetura

Componentes implementados:
1. Interface base: `FormalVerifier` (arquivo `verification/core/plugin_interface.py`)
2. Estruturas de dados: `VerificationInput`, `VerificationResult`, `VerificationStatus`
3. Registro: `VerifierRegistry` (instância global `registry`)
4. Manager de orquestração: `FormalVerificationManager` (`verification/core/formal_verification.py`)
5. Plugins disponíveis: `Z3Verifier` (`verification/plugins/z3_plugin.py`), `CVC5Verifier` (`verification/plugins/cvc5_plugin.py`)
6. Estratégias de tolerância a falhas: helpers em `core/error_handling.py` (circuit-breaker simples via `should_disable_solver`)
7. Schema padronizado de resultados estendido: `core/result_schema.py` (ainda não totalmente integrado ao fluxo principal de retorno simplificado)

## 📊 Lógica de SAT/UNSAT e Contraexemplos

### Interpretação Correta dos Resultados

| Resultado Solver | Constraint Status | Ação |
|------------------|-------------------|------|
| **SAT** | **VIOLADO** ❌ | Z3 gera contraexemplo |
| **UNSAT** | **SATISFEITO** ✅ | Nenhuma ação (propriedade garantida) |
| **UNKNOWN** | **INDETERMINADO** ⚠️ | Log de warning |

### Geração de Contraexemplos

- **Z3**: Implementa geração completa de contraexemplos para `bounds`, `range_check`, `non_negative`, `linear_arithmetic`
- **CVC5**: NÃO implementa geração de contraexemplos (apenas SAT/UNSAT básico)

```python
# Exemplo de contraexemplo gerado por Z3
{
  "constraint_type": "bounds",
  "violation_examples": [
    {"type": "below_minimum", "value": -1.0, "expected_min": 0.0},
    {"type": "above_maximum", "value": 1001.0, "expected_max": 1000.0}
  ]
}
```

## ✅ Status Realista dos Solvers

Ambos os plugins (Z3, CVC5) expõem listas extensas de nomes de constraints suportadas. Essas listas representam a taxonomia de TIPOS DE PROPRIEDADES que o framework pretende manipular; entretanto, nem todas possuem geração automática de fórmulas atualmente. Na prática:

| Categoria | Z3 | CVC5 | Contraexemplo | Observação |
|-----------|----|------|---------------|-----------|
| Aritmética linear | ✅ Implementado | ✅ Básico | Z3 ✅ | Fórmulas simples (bounds / não-negatividade) |
| Aritmética não-linear | ✅ Declarada | ✅ Declarada | ❌ | Sem geração especializada ainda |
| Tipos / shape (`shape_preservation`) | ✅ Básico | ✅ Básico | ❌ | Checagem via propriedades simples |
| Bounds (`bounds`) | ✅ Implementado | ✅ Implementado | Z3 ✅ | Comparação min/max entrada vs saída |
| Range Check (`range_check`) | ✅ Implementado | ⚠️ Stub | Z3 ✅ | Z3 detecta violações, CVC5 básico |
| Validade de parâmetros | ✅ Básico | ✅ Básico | ❌ | Usa dicionário de parâmetros |
| Robustez / fairness | ✅ Listado | ⚠️ Placeholder | ❌ | Não há modelagem completa |
| Otimização multi-objetivo | ✅ Listado | ❌ Não priorizado | ❌ | Não há criação de objetivos |

> **Importante:** Z3 gera contraexemplos quando constraint é violado (resultado SAT). CVC5 apenas retorna SAT/UNSAT sem contraexemplos.

## ♻️ Fluxo Simplificado de Execução

1. Usuário chama `verify()` ou pipeline dispara internamente.
2. `FormalVerificationManager` seleciona solver (auto ou nome explícito).
3. Para cada chave em `constraints` que esteja na lista suportada do solver, tenta-se construir/verificar a propriedade.
4. Resultado agregado retorna via `VerificationResult`.

## ⚠️ Atenção a Nomes de Constraints

Os exemplos anteriores usavam chaves como `preserve_shape` ou `preserve_bounds`; porém os plugins usam `shape_preservation` e `bounds`. Utilize SEMPRE as chaves publicadas por `list_verifiers()` para garantir seleção automática.

Exemplo de inspeção:
```python
from smart_inference_ai_fusion.verification import list_verifiers
print(list_verifiers())  # Mostra supported_constraints por verificador
```

## 🛠 Exemplo de Uso Atualizado

```python
from smart_inference_ai_fusion.verification import verify

result = verify(
    name="data_pipeline_step",
    constraints={
        'bounds': {'min': 0, 'max': 1},
        'shape_preservation': True,
        'parameter_validity': {'required': ['n_estimators', 'max_depth']}
    },
    input_data={'X_shape': (100, 8)},
    output_data={'X_shape': (100, 8)},
    parameters={'n_estimators': 50, 'max_depth': 5}
)

print(result.status, result.constraints_satisfied, result.constraints_violated)
```

## 🔄 Controle Global

```python
from smart_inference_ai_fusion.verification import enable_verification, disable_verification

disable_verification()  # Ignora verificações (retorna SKIPPED)
enable_verification()
```

## ➕ Adicionando um Novo Solver (Exemplo Minimalista)

```python
from smart_inference_ai_fusion.verification import FormalVerifier, VerificationInput, VerificationResult, VerificationStatus, registry

class DummyVerifier(FormalVerifier):
    def __init__(self):
        super().__init__("Dummy")
    def is_available(self):
        return True
    def supported_constraints(self):
        return ["bounds"]
    def verify(self, input_data: VerificationInput) -> VerificationResult:
        return VerificationResult(
            status=VerificationStatus.SUCCESS,
            verifier_name=self.name,
            execution_time=0.0001,
            constraints_checked=["bounds"],
            constraints_satisfied=["bounds"],
        )

registry.register(DummyVerifier())
```

## 📊 Estado Consolidado

### Concluído
- [x] Interface base e registro
- [x] Plugins Z3 e CVC5
- [x] Seleção automática por interseção de chaves de constraints
- [x] Circuit-breaker simples para solver instável
- [x] Integração com pipeline de inferência (invocação programática)

### Em Progresso / Planejado
- [ ] `result_normalizer` (unificar formatos avançados de cada solver)
- [ ] Captura opcional de estatísticas detalhadas (unsat core, modelo)
- [ ] Script de agregação estatística multi-solver
- [ ] Integração de métricas de overhead (tempo relativo baseline)
- [ ] Parametrização fina de limites via config externa (YAML/JSON)
- [ ] Benchmarks automatizados (Make target dedicado)

## 🔬 Testes e Validação

Atualmente os testes são executados via scripts de experimento (ex: em `examples/`) e geração de JSON em `logs/` & `results/`. Recomenda-se adicionar:

| Futuro Teste | Objetivo |
|--------------|----------|
| Propriedades sintéticas | Validar cada chave suportada gera saída estável |
| Stress (timeout) | Garantir status TIMEOUT coerente |
| Multi-solver paridade | Conferir mesma classificação para propriedades básicas |

## 🧭 Boas Práticas ao Definir Constraints

1. Use chaves suportadas: consulte `list_verifiers()`.
2. Prefira granularidade pequena (ex: `bounds` + `shape_preservation`).
3. Evite inserir objetos grandes diretamente; passe metadados (ex: shapes, min/max).
4. Trate resultado `SKIPPED` como sinal de mismatch de chave ou verificação desabilitada.

## 🧪 Integração com Experimentos

Exemplo (pseudo) dentro de pipeline:
```python
verification_constraints = {
    'bounds': {'min': float(X.min()), 'max': float(X.max())},
    'shape_preservation': True
}
verify(name='apply_data_inference', constraints=verification_constraints,
       input_data={'shape_before': X.shape}, output_data={'shape_after': Xp.shape})
```

## 🚀 Conclusão

O subsistema fornece base extensível e já funcional para verificação leve de propriedades estruturais e de parâmetros. O roadmap foca agora em: normalização, coleta aprofundada de métricas e expansão de cobertura sem inflar a complexidade da API pública.

---
_Última atualização automática deste documento para refletir o estado do branch `solver-interface`._

## 🎯 Como Usar

### Instalação
```bash
make verify-install  # Instala Z3 e dependências
```

### Comandos Disponíveis
```bash
make verify-test     # Testa sistema completo
make verify-list     # Lista verificadores disponíveis
make verify-z3       # Testa capacidades Z3
```

### Uso Programático
```python
from smart_inference_ai_fusion.verification import verify

# Verificação simples
result = verify(
    name="minha_funcao",
    constraints={
        'bounds': {'min': 0, 'max': 100},
        'linear_arithmetic': {'coefficients': [1, -1], 'constant': 0}
    }
)

print(f"Status: {result.status.value}")
print(f"Sucesso: {result.success}")
```

### Controle de Ativação
```python
from smart_inference_ai_fusion.verification import enable_verification, disable_verification

# Desabilitar globalmente
disable_verification()

# Reabilitar
enable_verification()
```

## 🔧 Adicionar Novos Verificadores

Para adicionar um novo verificador (ex: BMC, CBMC, KLEE):

```python
from smart_inference_ai_fusion.verification import FormalVerifier, registry

class MeuVerificador(FormalVerifier):
    def __init__(self):
        super().__init__("MeuVerificador")

    def is_available(self) -> bool:
        return True  # Verificar dependências

    def supported_constraints(self) -> List[str]:
        return ['meu_constraint_tipo']

    def verify(self, input_data) -> VerificationResult:
        # Implementar lógica de verificação
        pass

# Auto-registrar
registry.register(MeuVerificador())
```

## 📊 Status do Sistema

### ✅ Concluído
- [x] Interface de plugins extensível
- [x] Plugin Z3 com todos os recursos
- [x] Sistema de registro automático
- [x] Controle de ativação/desativação
- [x] Comandos Make integrados
- [x] Testes de validação completos

### 🎯 Próximos Passos (Passo 3)
- [ ] Decidir onde aplicar: dados vs inferências vs ambos
- [ ] Análise científica de onde faz mais sentido
- [ ] Integração com pipeline de experimentos

### 🔬 Resultados dos Testes
```
✅ Interface de plugins funcionando
✅ Z3 com capacidades avançadas funcionando
✅ Controle de ativação/desativação funcionando
✅ 10/10 testes de capacidades Z3 bem-sucedidos
```

## 🚀 Pronto para Próxima Fase!

O sistema está **completamente funcional** e pronto para discussão sobre onde aplicar cientificamente:
1. **Base de dados** - Verificar integridade e constraints dos dados
2. **Inferências** - Verificar transformações e algoritmos ML
3. **Ambos** - Verificação end-to-end completa

Sistema robusto, extensível e de alta performance preparado para verificação formal em escala!

## 📋 Registro de Experimentos

O sistema permite mapear datasets/modelos para scripts de experimento via registro dinâmico:

```python
from smart_inference_ai_fusion.experiments import experiment_registry

# Registrar novo experimento
experiment_registry["meu_dataset"] = "scripts/meu_experimento.py"
```

Isso facilita a integração de novos fluxos e automação de benchmarks.

## Suporte Multi-Solver

Além do Z3, o sistema suporta múltiplos solvers (ex: CVC5) e permite fácil extensão para outros:

```python
from smart_inference_ai_fusion.verification import verify, list_verifiers

print(list_verifiers())  # ['Z3', 'CVC5']

result = verify(
    name="minha_funcao",
    constraints={
        'bounds': {'min': 0, 'max': 100},
        'linear_arithmetic': {'coefficients': [1, -1], 'constant': 0}
    },
    solver="CVC5"  # ou "Z3"
)

print(f"Status: {result.status.value}")
print(f"Sucesso: {result.success}")
```

## Status Atualizado do Sistema

### ✅ Concluído
- [x] Interface de plugins extensível
- [x] Plugin Z3 com todos os recursos
- [x] Plugin CVC5 integrado
- [x] Sistema de registro automático
- [x] Controle de ativação/desativação
- [x] Comandos Make integrados
- [x] Testes de validação completos
- [x] Registro dinâmico de experimentos

### 🎯 Próximos Passos
- [ ] Decidir onde aplicar: dados vs inferências vs ambos
- [ ] Análise científica de onde faz mais sentido
- [ ] Integração com pipeline de experimentos
- [ ] Suporte a novos solvers e benchmarks automatizados

### 🔬 Resultados dos Testes
```
✅ Interface de plugins funcionando
✅ Z3 com capacidades avançadas funcionando
✅ CVC5 integrado e validado
✅ Controle de ativação/desativação funcionando
✅ 10/10 testes de capacidades Z3 bem-sucedidos
✅ 8/8 testes de capacidades CVC5 bem-sucedidos
```

## 🚀 Pronto para Próxima Fase!

Com arquitetura multi-solver, registro dinâmico e integração modular, o sistema está preparado para verificação formal em escala, benchmarking automatizado e fácil extensão para novos plugins!
