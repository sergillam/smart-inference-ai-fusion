# 📊 Técnicas de Inferência (Perturbação) Implementadas

Resumo do estado atual das perturbações no branch `solver-interface`. Cobrem três dimensões: dados (X), rótulos (y) e hiperparâmetros (params). O objetivo é gerar cenários de robustez / stress para análise e, opcionalmente, validação formal.

## 🧠 1. Perturbações em Dados (X)

Categorias principais: ruído, estrutura, precisão, corrupção, outliers e distração.

| Arquivo (atual) | Categoria | Técnicas (exemplos) | Efeito | Observação |
|-----------------|----------|---------------------|--------|-----------|
| `inference/transformations/data/noise.py` | Ruído | `GaussianNoise`, `FeatureSelectiveNoise` | Suaviza fronteiras / altera atributos específicos | Seleção de features deve respeitar dimensionalidade (bug anterior resolvido com configs por dataset) |
| `inference/transformations/data/precision.py` | Precisão | `TruncateDecimals`, `CastToInt`, `Quantize` | Reduz granularidade | Útil para simular compressão / dispositivos limitados |
| `inference/transformations/data/structure.py` | Estrutura | `ShuffleFeatures`, `ScaleFeatures`, `RemoveFeatures`, `FeatureSwap` | Rompe correlações ou remove sinais | `RemoveFeatures` exige cuidado com índices |
| `inference/transformations/data/corruption.py` | Corrupção | `ZeroOut`, `InsertNaN` | Perda parcial/total de informação | Pode acionar verificação de bounds │
| `inference/transformations/data/outliers.py` | Outliers | `InjectOutliers` | Distorce distribuição | Parâmetros controlam fração/intensidade |
| `inference/transformations/data/distraction.py` | Distração | `AddDummyFeatures`, `DuplicateFeatures` | Aumenta dimensionalidade irrelevante | Pode testar robustez a overfitting |
| `inference/transformations/data/cluster_swap.py` | Estrutura | `ClusterSwap` | Mistura clusters entre si | Stress de separabilidade |
| `inference/transformations/data/group_outlier_injection.py` | Outliers direcionados | `GroupOutlierInjection` | Injeta outliers condicionados a grupo | Simula ataque direcionado |
| `inference/transformations/data/conditional_noise.py` | Ruído condicional | `ConditionalNoise` | Ruído aplicado sob condição semântica | Requer metadados auxiliares |

## 🧪 2. Perturbações em Rótulos (y)

| Arquivo | Técnicas | Efeito |
|---------|----------|--------|
| `inference/transformations/label/label_noise.py` | `LabelNoise` (e variantes internas) | Simula erro de anotação / barulho sistemático |

Extensões futuras podem incluir: confusão parcial entre classes minoritárias, inversão dirigida por distribuição, preservação de entropia.

## 🛠️ 3. Perturbações em Parâmetros (Hiperparâmetros)

Local: `inference/transformations/params/`

| Arquivo | Tipo | Objetivo |
|---------|------|----------|
| `int_noise.py` | Numérico | Incrementos / desvios controlados (inteiros) |
| `scale_hyper.py` | Numérico | Escalar hiperparâmetros contínuos |
| `type_cast_perturbation.py` | Tipo | Casting forçado (int→str, float→str) |
| `enum_boundary_shift.py` | Enum | Empurrar valores para limites válidos/alternativos |
| `bool_flip.py` | Booleano | Inversão lógica |
| `semantic_mutation.py` | Semântico | Mutação textual (`rbf`→`rbf_x`) |
| `str_mutator.py` | String | Inserção / alteração de tokens |
| `bounded_numeric.py` | Numérico | Garante (ou viola) limites configurados |
| `random_from_space.py` | Espaço | Seleção aleatória coerente com search space |
| `cross_dependency.py` | Dependência cruzada | Ajustes condicionais entre parâmetros |
| `base.py` | Base | Classes utilitárias / abstrações |

Essas compõem o núcleo da noção de "SmartParameterPerturber" (conceito distribuído nos módulos). Estratégias combinadas permitem cenários como: ligeiro drift + casting inválido + enum perturbado.


### Estratégias Sintéticas (Visão Abstrata)

| Tipo do Parâmetro | Ações (exemplos) | Exemplo | Efeito |
|-------------------|------------------|---------|--------|
| `int` | ruído discreto, cast, drop | 3 → 4 / "3" / None | Avalia tolerância a limites |
| `float` | ruído gaussiano, cast, drop | 1.0 → 1.3 / "1.0" | Variação de precisão |
| `str` | mutação semântica | "rbf" → "rbf_x" | Testa validação interna |
| `bool` | flip / cast | True → False / "True" | Verifica handling lógico |
| `enum` | boundary shift | 'gini' → 'entropy' | Explora alternativas válidas |
| `NoneType` | substituição | None → 0 | Força inicialização |

> Observação: algumas ações resultam em parâmetros potencialmente inválidos — combinável com constraint `parameter_validity` em verificação.

## 🤝 Integração com Verificação Formal

Após cada bloco de perturbação o pipeline pode (condicionalmente) chamar `verify()` para:
- `bounds` / `shape_preservation` (dados)
- `parameter_validity` / `data_shape_validation` (parâmetros)

Isso cria laços de feedback para detectar transformações destrutivas ou inconsistentes.


## 🚀 Pipeline e Registro de Experimentos

O pipeline de inferência permite aplicar múltiplas técnicas de perturbação em dados, rótulos e parâmetros, integrando com o registro dinâmico de experimentos:

```python
from smart_inference_ai_fusion.verification import verify
from smart_inference_ai_fusion.experiments.experiment_registry import run_experiment_by_model
from smart_inference_ai_fusion.utils.types import SklearnDatasetName
from smart_inference_ai_fusion.models.random_forest_classifier_model import RandomForestClassifierModel

# Executa experimento registrado (baseline + perturbações + opcional verificação)
baseline, inference = run_experiment_by_model(
	model_class=RandomForestClassifierModel,
	dataset_name=SklearnDatasetName.MAKE_BLOBS,
)

# Exemplo de chamada explícita de verificação pós-perturbação
verify(
	name="apply_data_inference",
	constraints={'bounds': {'min': 0, 'max': 10}, 'shape_preservation': True}
)
```

## ➕ Adicionando Novas Técnicas (Dados / Parâmetros / Rótulos)

Para estender o framework, basta criar uma nova função de perturbação e registrá-la como plugin:

```python
"""
Seguir padrões existentes:
 - Para dados: criar função/classe em `inference/transformations/data/` e adicionar ao builder do engine.
 - Para parâmetros: adicionar transformação em `inference/transformations/params/` herdando de base apropriada.
 - Para rótulos: estender `label_noise.py` ou criar novo módulo.
"""
```

## 🔌 Multi-Solver (Verificação)

O sistema de inferência é agnóstico a solver; a verificação formal (Z3/CVC5) opera sobre metadados pós-perturbação.

## 🧭 Roadmap de Melhoria

| Item | Status | Comentário |
|------|--------|------------|
| Normalização central de resultados de perturbação | Planejado | Facilitar comparação cruzada |
| Métricas agregadas de impacto (ex: drift por transformação) | Planejado | Alimentar relatórios científicos |
| Seleção adaptativa de perturbações (feedback verification) | Ideia | Usar violações para explorar vizinhança |
| Testes unitários por transformação | Parcial | Ampliar cobertura edge cases |

## 📋 Resumo Final

O subsistema de inferência suporta ampla gama de perturbações componíveis em dados, rótulos e hiperparâmetros, servindo de base para avaliações de robustez e integração com verificação formal multi-solver.