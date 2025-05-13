# 📊 Técnicas de Inferência Implementadas no Framework

Este documento resume todas as técnicas de inferência sintética atualmente implementadas no framework `smart-inference-ai-fusion`, organizadas em duas grandes categorias: transformações nos dados de entrada (X, y) e nos parâmetros dos modelos (kwargs).

## 🧠 1. Inferência Aplicada ao Dataset (X)

Essas técnicas simulam ruídos realistas, falhas de sensores, perturbações estruturais e corrupções diretas nos dados.

| Arquivo          | Categoria              | Técnica               | Impacto Esperado                                    |
| ---------------- | ---------------------- | --------------------- | --------------------------------------------------- |
| `src/inference/noise.py`       | Ruído Aditivo          | GaussianNoise         | Desestabiliza decisões próximas de fronteiras       |
| `src/inference/noise.py`       | Ruído Aditivo          | FeatureSelectiveNoise | Ruído apenas em atributos específicos               |
| `src/inference/precision.py`   | Redução de Precisão    | TruncateDecimals      | Perda de distinção entre pontos próximos            |
| `src/inference/precision.py`   | Redução de Precisão    | CastToInt             | Simplifica variações contínuas                      |
| `src/inference/precision.py`   | Redução de Precisão    | Quantize              | Discretiza os dados com bins fixos                  |
| `src/inference/structure.py`   | Perturbação Estrutural | ShuffleFeatures       | Quebra correlação entre colunas                     |
| `src/inference/structure.py`   | Perturbação Estrutural | ScaleFeatures         | Altera magnitude entre atributos                    |
| `src/inference/structure.py`   | Perturbação Estrutural | RemoveFeatures        | Remove atributos simulando sensores com falha       |
| `src/inference/structure.py`   | Perturbação Estrutural | FeatureSwap           | Troca valores entre amostras                        |
| `src/inference/corruption.py`  | Corrupção Direta       | ZeroOut               | Apaga parcialmente os dados                         |
| `src/inference/corruption.py`  | Corrupção Direta       | InsertNaN             | Simula leitura com falha completa                   |
| `src/inference/outliers.py`    | Perturbação Extrema    | InjectOutliers        | Injeta valores extremos que distorcem distribuições |
| `src/inference/distraction.py` | Distração Semântica    | AddDummyFeatures      | Atributos irrelevantes confundem o modelo           |
| `src/inference/distraction.py` | Distração Semântica    | DuplicateFeatures     | Colunas redundantes que aumentam dimensionalidade   |
| `src/inference/label_noise.py` | Rótulos Corrompidos    | LabelNoise            | Rótulos trocados simulam erro de anotação           |


# 🧩 2. Inferência Aplicada aos Parâmetros dos Modelos

Essas estratégias simulam erros de configuração, variações inesperadas, ou entradas semânticas incorretas nos hiperparâmetros dos modelos.

## Estratégias aplicadas (`SmartParameterPerturber`)

| Tipo do Parâmetro | Estratégias                              | Exemplo                           |
| ----------------- | ---------------------------------------- | --------------------------------- |
| `int`             | add\_noise, cast\_str, cast\_float, drop | 3 → 4.0, "3", None                |
| `float`           | add\_noise, cast\_str, drop              | 1.0 → 1.3, "1.0", None            |
| `str`             | mutate, drop, cast                       | "rbf" → "invalid", None, "rbf\_x" |
| `bool`            | flip, cast\_str, drop                    | True → False, "True", None        |
| `NoneType`        | replace\_none                            | None → 0                          |

## Recursos Inteligentes (v1.0)

- ✅ Autoidentificação de tipo dos parâmetros
- ✅ Fallback automático se a perturbação causar erro
- ✅ Exclusão de parâmetros obrigatórios conhecidos (ex: `'C'`, `'kernel'`)
- ✅ Memoização leve para evitar repetições
- ✅ Exportação de log completo via `export_log()`