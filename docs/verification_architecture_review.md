"""Revisão Arquitetural Atualizada – Subsistema de Verificação Formal"""

## 1. Visão Geral

O subsistema de verificação formal fornece uma camada plugin-based para verificar propriedades declaradas pelo usuário após transformações de inferência (dados, labels, parâmetros). Dois solvers SMT estão integrados: Z3 e CVC5. O design privilegia extensibilidade com retorno simplificado (`VerificationResult`).

## 2. Pontos Fortes (Estado Atual)

| Área | Força | Evidência (arquivo) |
|------|-------|---------------------|
| Interface clara | `FormalVerifier` com métodos mínimos (`is_available`, `supported_constraints`, `verify`) | `core/plugin_interface.py` |
| Registro central | Instância global `registry` acessível via import | `core/plugin_interface.py` |
| Multi-solver | Plugins Z3 + CVC5 com configuração de alto desempenho | `plugins/z3_plugin.py`, `plugins/cvc5_plugin.py` |
| Fallback / resiliência | Funções `handle_verification_error` e `should_disable_solver` | `core/error_handling.py` |
| Riqueza taxonômica | Listas extensas de constraint names para futura expansão | Ambos plugins |
| Integração leve | Fácil chamada via `verify()` em pipelines | `core/formal_verification.py` |

## 3. Limitações / Gaps

| Tipo | Descrição | Impacto | Prioridade |
|------|-----------|---------|-----------|
| Cobertura real | Muitas constraints listadas não possuem geração de fórmulas concreta | Risco de falsa expectativa | Alta |
| Normalização | `result_schema.py` não integrado a retorno padrão | Perda de granularidade estruturada | Média |
| Métricas | Sem coleta sistemática de tempo incremental vs baseline | Difícil mensurar overhead | Alta |
| Multi-solver comparativo | Não há módulo de consolidação/paridade automatizada | Não mede divergências | Alta |
| Config externa | Constraints só via dicionário inline | Dificulta replicabilidade | Média |
| Testes unitários | Ausência de suite dedicada para cada constraint básica | Menor confiança em regressões | Alta |
| Logs estruturados | Logs informativos porém não segmentados por ID de verificação | Análise batch limitada | Média |

## 4. Oportunidades de Evolução

1. Normalizador de Resultado (`result_normalizer`): converte `VerificationResult` + artefatos solver → schema padronizado (JSON serializável) armazenado em `results/verification/`.
2. Mecanismo de Paridade Multi-solver: compara sets `(constraints_satisfied, constraints_violated)` e produz métricas: Jaccard, conflito crítico, tempo relativo.
3. Perfilador de Overhead: wrapper que mede (t_execução_sem_verificação, t_com_verificação) e grava delta.
4. Catálogo de Constraints Documentado: tabela Markdown gerada automaticamente enumerando exemplos de payload esperado por chave.
5. Test Harness: mini fábrica de cenários (synthetic arrays) para cada constraint suportada basal (`bounds`, `shape_preservation`, `parameter_validity`).

## 5. Recomendações Concretas (Backlog Priorizado)

| Ordem | Tarefa | Definição de Pronto |
|-------|--------|---------------------|
| 1 | Implementar `result_normalizer.py` | Função `normalize(VerificationResult)->dict` + teste |
| 2 | Criar `multi_solver_compare.py` | Métrica Jaccard + JSON de divergências |
| 3 | Adicionar coleta de tempo (%) | Campo `details['overhead_ms']` calculado no manager |
| 4 | Suite mínima de testes constraints | 3 testes: bounds ok, bounds violado, shape preservado |
| 5 | Export de catálogo (`make verify-doc`) | Gera `docs/constraints_catalog.md` |
| 6 | Suporte a config YAML | Carrega constraints de arquivo externo |

## 6. Modelo de Dados Proposto (Normalização)

```jsonc
{
  "solver": "Z3",
  "status": "success|failure|error|timeout|skipped",
  "timing_ms": 12.4,
  "constraints": {
    "checked": ["bounds", "shape_preservation"],
    "satisfied": ["bounds", "shape_preservation"],
    "violated": []
  },
  "meta": {
    "input_shape_before": [100, 8],
    "input_shape_after": [100, 8],
    "parameters_checked": ["n_estimators"],
    "overhead_ms": 3.2
  }
}
```

## 7. Critérios de Qualidade Futuro (Definition of Done Extendida)

| Critério | Meta |
|----------|------|
| Precisão básica | 100% detecção de violação artificial de bounds |
| Estabilidade | Nenhum crash em 100 execuções consecutivas com constraints triviais |
| Overhead | < 10% de acréscimo médio para constraints básicas em datasets < 10k instâncias |
| Paridade básica | Z3 e CVC5 concordam em >= 95% dos casos simples |

## 8. Riscos e Mitigações

| Risco | Mitigação |
|-------|----------|
| Lista inflada de constraints sem suporte real | Gerar catálogo versionado marcando status (implementado / placeholder) |
| Overhead crescente com múltiplos solvers | Modo configurável: `verification_mode = first_success | all | disabled` |
| Inconsistência de logs | Adicionar identificador UUID por chamada de verificação |

## 9. Próximos Passos Imediatos

1. Implementar normalização + comparação (iteração curta).
2. Adicionar medição de overhead no `FormalVerificationManager` (wrapper de tempo antes/depois).
3. Criar testes unitários mínimos para constraints centrais.

## 10. Conclusão

Arquitetura estável e extensível já suporta múltiplos solvers e oferece interface uniforme. Evoluções agora devem focar em: (i) concretização de semantics para constraints listadas, (ii) instrumentação e medição reprodutível, (iii) transparência documental (catálogo). Isso habilitará análises científicas robustas (RQ1–RQ3) com base em evidências empíricas e comparáveis.

---
_Documento revisado automaticamente para refletir estado real do código em `solver-interface`._
