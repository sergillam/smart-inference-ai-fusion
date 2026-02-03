# Veredictos de SMT Solvers

Este documento descreve os veredictos (outputs) típicos dos SMT solvers integrados no projeto (Z3, CVC5), o esquema de normalização adotado pela camada de verificação e exemplos JSON de entradas/saídas.

## Objetivo

1. Padronizar os campos de saída dos plugins (`z3_plugin`, `cvc5_plugin`) para que o `VerificationManager` possa comparar e agregar resultados.
2. Fornecer exemplos de veredictos brutos e normalizados para facilitar debugging e escrita de análises (RQ experiments).

## Status comuns

- `sat` — existe modelo que satisfaz as restrições.
- `unsat` — conjunto de restrições é insatisfatável.
- `unknown` — solver não conseguiu concluir (timeout, heurísticas).
- `error` — falha na execução do solver (sintaxe, recurso não suportado, crash).

Observação: alguns solvers podem devolver variantes (e.g. `unsat-with-core`); normalize para os quatro estados acima quando possível.

## Esquema de saída normalizado (VerificationResult)

Estrutura JSON/Dataclass usada internamente:

```json
{
  "solver": "z3",
  "status": "unsat",                
  "time_ms": 123,                      
  "model": null,                       
  "unsat_core": ["c1","c2"],       
  "num_constraints": 42,               
  "matched_constraints": [0,1,2],     
  "raw_output": "<texto bruto do solver>",
  "error": null
}
```

Campos explicados:
- `solver`: identificador do solver (e.g. `z3`, `cvc5`).
- `status`: um dos `sat`, `unsat`, `unknown`, `error`.
- `time_ms`: tempo de execução medido (milissegundos), quando disponível.
- `model`: objeto/representação do modelo se `sat` (pode ser um dicionário com valores das variáveis ou string bruta). `null` caso contrário.
- `unsat_core`: lista de rótulos/ids de restrições que compõem o núcleo insatisfatível (quando fornecido pelo solver).
- `num_constraints`: número total de restrições submetidas.
- `matched_constraints`: índices/ids das restrições que o solver reconheceu ou que foram usadas para produzir `unsat_core`.
- `raw_output`: saída textual crua do solver (útil para auditoria e debugging).
- `error`: objeto/string com mensagem de erro quando `status` == `error`.

## Exemplo — saída bruta (Z3)

Exemplo simplificado (mocked):

```text
sat
(model
  (define-fun x () Int 5)
  (define-fun y () Int 3)
)
```

O plugin Z3 deve capturar esse texto, medir tempo e produzir o `VerificationResult` normalizado.

## Exemplo — saída bruta (CVC5)

```text
unknown
reason: timeout
```

Nesse caso o plugin deve mapear para `status: "unknown"` e colocar `error`/`raw_output` com a explicação.

## Exemplos normalizados (Z3 e CVC5)

Z3 — modelo sat:

```json
{
  "solver": "z3",
  "status": "sat",
  "time_ms": 85,
  "model": {"x": 5, "y": 3},
  "unsat_core": null,
  "num_constraints": 3,
  "matched_constraints": [0,1,2],
  "raw_output": "sat\n(model\n  (define-fun x () Int 5)\n  (define-fun y () Int 3)\n)",
  "error": null
}
```

CVC5 — timeout/unknown:

```json
{
  "solver": "cvc5",
  "status": "unknown",
  "time_ms": 5000,
  "model": null,
  "unsat_core": null,
  "num_constraints": 3,
  "matched_constraints": [0,1],
  "raw_output": "unknown\nreason: timeout",
  "error": "timeout"
}
```

## Como o `VerificationManager` usa estes veredictos

- Recebe `VerificationResult` de cada plugin.
- Normaliza os campos (padronizando `status`, extraindo `unsat_core` quando possível).
- Calcula métricas comparativas:
  - paridade (both unsat / both sat / disagreement),
  - tempo relativo e overhead,
  - cobertura de constraints (quantas constraints aparecem no `unsat_core`).
- Armazena ambos os resultados brutos e normalizados em `logs/` e `results/` para auditoria.

## Boas práticas para plugins

- Sempre preencher `raw_output` com a saída textual do solver.
- Medir e preencher `time_ms` com precisão.
- Quando possível, mapear nomes/ids das constraints para índices estáveis (ex.: `c0`, `c1`, ...) antes de enviar ao solver para que `unsat_core` seja interpretável.
- Em caso de `error`, preencher `error` com um objeto detalhado {code, message, hint} para facilitar triagem.

## Uso em experimentos (RQ)

- Registre, por experimento, os veredictos de cada solver junto com seed do dataset e versão do solver (ex.: `z3 4.8.17`, `cvc5 1.0`).
- Compare percentuais de concordância (unsat vs sat), tempo médio, e tamanho médio do `unsat_core`.

## Próximos passos recomendados

- Implementar/validar a função `result_normalizer` no `verification` para garantir que cada plugin gere exatamente o esquema acima.
- Adicionar testes unitários que simulam saídas brutas de Z3/CVC5 e verificam a normalização.
- Incluir versão do solver no `VerificationResult` (campo `solver_version`) para reprodutibilidade.

---

Arquivo criado automaticamente para documentar os veredictos; posso também:
- gerar exemplos a partir de execuções reais (se `z3`/`cvc5` estiverem instalados),
- adicionar um gerador de fixtures JSON em `tests/fixtures/verification_results/`.

## Campos detalhados do `result_schema.py`

Para facilitar a análise e inclusão no artigo, abaixo está uma tabela com os campos principais das duas estruturas centrais: `SolverMetadata` e `PerformanceMetrics` (descritos em `verification/core/result_schema.py`).

### SolverMetadata (fields)

- `solver_name` (str): identificador do solver (ex.: `z3`, `cvc5`).
- `solver_version` (str): versão do solver (ex.: `4.8.17`).
- `logic_used` (str): lógica SMT usada (ex.: `QF_NIRA`).
- `timeout_ms` (int): timeout configurado para a invocação (ms).
- `memory_limit_mb` (int): limite de memória configurado (MB).
- `thread_count` (int): número de threads alocadas para o solver.
- `random_seed` (Optional[int]): seed usado para determinismo (quando aplicável).
- `configuration_hash` (Optional[str]): hash/identificador da configuração usada.
- `system_info` (Optional[Dict[str,str]]): informações do sistema (platform, processor, cpu_count, python_version).

### PerformanceMetrics (fields)

- `total_execution_time` (float): tempo total de execução do solver (segundos).
- `constraint_count` (int): número total de constraints submetidas.
- `constraints_satisfied` (int): quantas constraints foram reportadas como satisfeitas.
- `constraints_violated` (int): quantas constraints foram reportadas como violadas.
- `constraints_unknown` (int): quantas ficaram em estado `unknown`.
- `constraints_timeout` (int): quantas terminaram em timeout.
- `constraints_error` (int): quantas terminaram com erro.
- `constraints_skipped` (int): quantas foram puladas.
- `memory_usage_mb` (Optional[float]): uso médio de memória (MB), quando disponível.
- `peak_memory_mb` (Optional[float]): pico de uso de memória (MB), quando disponível.
- `restart_count` (Optional[int]): número de restarts do solver (quando esse dado é exposto).
- `decisions_count` (Optional[int]): contagem de decisões (se exposta pelo solver).
- `propagations_count` (Optional[int]): contagem de propagations (se exposta pelo solver).
- `conflicts_count` (Optional[int]): número de conflitos.

### Derivadas/Propriedades calculadas

- `success_rate`: calculado como `constraints_satisfied / constraint_count`.
- `completion_rate`: calculado como `(satisfied + violated + unknown) / constraint_count`.

## Inclusão no workflow experimental

- Ao armazenar `StandardVerificationResult.to_json()` em `logs/`, garanta que `solver_metadata` e `performance` estejam preenchidos para permitir análises comparativas e reprodutibilidade.
- Recomenda-se salvar também um `experiment_manifest.json` com: timestamp, seed, solver versions, environment (docker image / system_info), e o commit hash do repositório.

