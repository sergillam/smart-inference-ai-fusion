# Mapeamento dos Verificadores Formais - Z3 e CVC5

Este documento descreve o mapeamento detalhado da implementação dos verificadores formais Z3 e CVC5, incluindo a lógica de SAT/UNSAT e geração de contraexemplos.

## 📊 Resumo da Lógica de Verificação

### Resultado do Solver vs. Resultado da Verificação

| Resultado Solver | Constraint Status | Ação | Descrição |
|------------------|-------------------|------|-----------|
| **SAT** | **VIOLADO** ❌ | Gera **contraexemplo** | O solver encontrou valores que satisfazem a negação do constraint → violação detectada |
| **UNSAT** | **SATISFEITO** ✅ | Nenhuma ação | Não existe valor que viole o constraint → propriedade garantida |
| **UNKNOWN** | **INDETERMINADO** ⚠️ | Log de warning | Solver não conseguiu decidir (timeout, complexidade) |

### ⚠️ IMPORTANTE: Interpretação Correta

A lógica de verificação formal usa **negação**:
- Para verificar se `x ∈ [min, max]`, o solver testa se `∃x: x < min ∨ x > max`
- Se encontrar (SAT) → violação existe → constraint **VIOLADO**
- Se não encontrar (UNSAT) → nenhuma violação possível → constraint **SATISFEITO**

---

## 🔍 Z3 Verifier (`z3_plugin.py`)

### Arquitetura

```
Z3Verifier
├── _init_z3()                          # Configuração de alto desempenho
├── verify()                            # Entrada principal
├── _verify_constraint_with_details()   # Verificação + coleta de detalhes
├── _verify_constraint()                # Dispatch para métodos específicos
├── _generate_counterexample()          # Geração de contraexemplos
└── _report_verification_details()      # Reporting (console, JSON)
```

### Configuração do Solver

```python
# Lógica: QF_NIRA (Quantifier-free Nonlinear Integer Real Arithmetic)
# Timeout: 600000ms (10 minutos)
# Memória: 14GB
# Threads: 16 (ou os.cpu_count())
# Seed: 12345 (determinístico)
```

### Lógica de Verificação por Constraint

#### 1. Bounds (`bounds`)

```python
def _verify_bounds(constraint_data, input_data):
    """
    Verifica: ∀x ∈ data: min ≤ x ≤ max

    Implementação:
    - Para cada valor em data_array:
      1. solver.push()
      2. solver.add(x == valor)
      3. Se solver.check() != SAT → violação
      4. solver.pop()

    Resultado:
    - satisfied=True: todos valores dentro dos bounds
    - satisfied=False: pelo menos um valor fora
    """
```

**Contraexemplo gerado:**
```json
{
  "constraint_type": "bounds",
  "violation_examples": [
    {"type": "below_minimum", "value": -1.0, "expected_min": 0.0},
    {"type": "above_maximum", "value": 1001.0, "expected_max": 1000.0}
  ]
}
```

#### 2. Range Check (`range_check`)

```python
def _verify_range_check(constraint_data):
    """
    Verifica: valor ∈ valid_ranges (contínuo ou discreto)

    Tipos:
    - continuous: OR de intervalos [min_i, max_i]
    - discrete: OR de valores específicos
    """
```

#### 3. Type Safety (`type_safety`)

```python
def _verify_constraint():
    """
    Type safety é verificado estruturalmente.
    Sempre retorna True se o tipo está correto.
    """
```

#### 4. Real Arithmetic (`real_arithmetic`)

```python
def _verify_constraint():
    """
    Verifica propriedades de aritmética real.
    Exemplo: operações preservam tipo real.
    """
```

### Fluxo de Geração de Contraexemplos

```python
def _verify_constraint_with_details():
    # 1. Executa verificação
    satisfied = self._verify_constraint(...)

    # 2. Coleta resultado SAT/UNSAT
    check_result = self.solver.check()

    # 3. Se SAT → captura modelo
    if check_result == z3.sat:
        model = self.solver.model()
        details["z3_model"] = model

    # 4. Se constraint violado → gera contraexemplo
    if not satisfied:
        counterexample = self._generate_counterexample(...)
        details["counterexample"] = counterexample
```

### Métodos de Contraexemplo

| Constraint | Método | Gera |
|------------|--------|------|
| `bounds` | `_generate_bounds_counterexample()` | Valores fora de [min, max] |
| `range_check` | `_generate_range_counterexample()` | Valores fora dos ranges válidos |
| `non_negative` | `_generate_non_negative_counterexample()` | Valores negativos |
| `linear_arithmetic` | `_generate_linear_arithmetic_counterexample()` | Violações de equações lineares |

---

## 🔧 CVC5 Verifier (`cvc5_plugin.py`)

### Arquitetura

```
CVC5Verifier
├── _init_cvc5()                    # Configuração científica
├── verify()                        # Entrada principal + reporting
├── _verify_constraint()            # Dispatch para métodos específicos
└── _verify_*_constraint()          # Métodos específicos por tipo
```

### Configuração do Solver

```python
# Lógica: QF_NIRA
# Timeout: 600000ms (10 minutos)
# Opções especiais:
#   - nl-ext: true (extensões não-lineares)
#   - nl-cad: true (Cylindrical Algebraic Decomposition)
#   - produce-models: true
#   - produce-unsat-cores: true
```

### Lógica de Verificação

#### Diferença Chave do CVC5

O CVC5 implementa verificação mais simplificada:

```python
def _verify_bounds_constraint(constraint_value, input_data):
    """
    CVC5 verifica bounds de forma direta:
    1. Cria variável real x
    2. Adiciona constraints: x >= min AND x <= max
    3. Verifica SAT

    Resultado:
    - SAT → bounds são satisfazíveis
    - UNSAT → bounds impossíveis (ex: min > max)
    """

    result = self.solver.checkSat()
    is_sat = result.isSat()

    return {
        "satisfied": is_sat,
        "cvc5_result": str(result),
        "cvc5_satisfiable": is_sat
    }
```

### ⚠️ CVC5 NÃO Gera Contraexemplos Automaticamente

~~Atualmente, o CVC5 plugin **não implementa geração de contraexemplos**. Apenas retorna:~~

✅ **ATUALIZADO (28/11/2025):** O CVC5 plugin agora implementa:
- Verificação real dos dados contra bounds (não apenas satisfatibilidade)
- Geração de contraexemplos quando constraint é violado (SAT na negação)
- Mesma lógica do Z3: SAT → violação encontrada → constraint VIOLADO

Retorna:
- `satisfied: True/False` - baseado na verificação real dos dados
- `cvc5_result: "sat"/"unsat"` - SAT significa violação encontrada
- `counterexample: {...}` - quando constraint é violado, inclui exemplos de violação
- `details: string descritiva`

---

## 📋 Tabela Comparativa Z3 vs CVC5

| Feature | Z3 | CVC5 |
|---------|-----|------|
| **Lógica Padrão** | QF_NIRA | QF_NIRA |
| **Timeout** | 10 min | 10 min |
| **Memória** | 14GB | 12GB |
| **Contraexemplos** | ✅ Sim | ✅ Sim (atualizado) |
| **Modelo (SAT)** | ✅ Captura | ✅ Captura (atualizado) |
| **UNSAT Core** | ✅ Captura | ⚠️ Configurado |
| **Estatísticas** | ✅ Detalhadas | ⚠️ Básicas |
| **Constraints Complexos** | ✅ Muitos | ✅ Básicos + bounds/non_negative |

---

## 📁 Estrutura de Arquivos de Resultado

### Z3 (`results/z3-verification-*.json`)

```json
{
  "verification_session": {
    "verifier": "Z3",
    "timestamp": "pipeline.parameters.pre_perturbation",
    "execution_time_ms": 8.57,
    "success_rate": 75.0
  },
  "constraint_results": {
    "satisfied": ["type_safety", "range_check", "real_arithmetic"],
    "violated": ["bounds"]
  },
  "z3_solver_details": {
    "bounds": {
      "z3_result": "sat",
      "satisfied": false,
      "counterexample": {
        "violation_examples": [
          {"type": "below_minimum", "value": -1.0},
          {"type": "above_maximum", "value": 1001.0}
        ]
      }
    }
  }
}
```

### CVC5 (`results/cvc5-verification-*.json`)

```json
{
  "verification_session": {
    "verifier": "CVC5",
    "timestamp": "pipeline.data.input_data",
    "execution_time_ms": 5.23,
    "success_rate": 100.0
  },
  "constraint_results": {
    "satisfied": ["shape_preservation", "type_safety", "bounds"],
    "violated": []
  },
  "cvc5_solver_details": {
    "bounds": {
      "satisfied": true,
      "cvc5_result": "sat",
      "details": "Bounds constraint verified"
    }
  }
}
```

---

## 🎯 Constraints Implementados

### Z3 - Completo

| Constraint | Implementado | Contraexemplo |
|------------|--------------|---------------|
| `bounds` | ✅ | ✅ |
| `range_check` | ✅ | ✅ |
| `type_safety` | ✅ | ❌ |
| `non_negative` | ✅ | ✅ |
| `real_arithmetic` | ✅ | ❌ |
| `linear_arithmetic` | ✅ | ✅ |
| `shape_preservation` | ⚠️ Básico | ❌ |
| `invariant` | ✅ | ❌ |
| `precondition` | ✅ | ❌ |
| `postcondition` | ✅ | ❌ |
| `robustness` | ✅ | ❌ |

### CVC5 - Completo (Atualizado 2025-01-XX)

| Constraint | Implementado | Contraexemplo | Prioridade |
|------------|--------------|---------------|------------|
| `bounds` | ✅ | ✅ | Alta |
| `range_check` | ✅ | ✅ | Alta |
| `type_safety` | ⚠️ Stub | ❌ | Baixa |
| `non_negative` | ✅ | ✅ | Alta |
| `shape_preservation` | ⚠️ Stub | ❌ | Baixa |
| `linear_arithmetic` | ✅ | ✅ | Alta |
| `real_arithmetic` | ✅ | ✅ | Média |
| `integer_arithmetic` | ✅ | ✅ | Média |
| `floating_point` | ✅ | ✅ | Alta |
| `invariant` | ✅ | ✅ | Alta |
| `precondition` | ✅ | ✅ | Alta |
| `postcondition` | ✅ | ✅ | Alta |
| `robustness` | ✅ | ✅ | Alta |
| `parameter_drift` | 🔧 Básico | ❌ | Baixa |
| `model_instantiation` | 🔧 Básico | ❌ | Baixa |
| `parameter_consistency` | 🔧 Básico | ❌ | Baixa |
| `attribute_check` | 🔧 Básico | ❌ | Baixa |

**Resumo CVC5:**
- Total: 17 métodos
- Implementação completa com contraexemplos: 11 (65%)
- Implementação básica/SMT: 4 (23%)
- Stubs: 2 (12%)

---

## 🔄 Fluxo de Orquestração (FormalVerificationManager)

```
1. verify() chamado
   ↓
2. Seleciona solver (auto ou específico)
   ↓
3. Para cada constraint em input.constraints:
   │
   ├─→ Z3: _verify_constraint_with_details()
   │       → Verifica
   │       → Coleta modelo/core
   │       → Gera contraexemplo se violado
   │
   └─→ CVC5: _verify_constraint()
             → Verifica
             → Retorna SAT/UNSAT básico
   ↓
4. Agrega resultados
   ↓
5. Gera VerificationResult
   ↓
6. Salva em logs/ e results/
```

---

## 📊 Estatísticas dos Logs (Exemplo Real)

Da análise de `logs/` e `results/`:

| Property | CVC5 SAT | CVC5 UNSAT | Z3 SAT | Z3 UNSAT |
|----------|----------|------------|--------|----------|
| Bounds | 151 | 0 | 74 | 0 |
| Range Check | 151 | 0 | 37 | 37 |
| Type Safety | 151 | 0 | 111 | 0 |
| Shape Preservation | 151 | 0 | 0 | 0 |
| Real Arithmetic | 0 | 0 | 74 | 0 |
| **Total** | **678** | **0** | **296** | **37** |

> **Nota:** Z3 detectou 37 violações em `range_check` que CVC5 não detectou (diferença de bounds herdados).
