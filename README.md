# smart-inference-ai-fusion

#### Framework modular e extensível para experimentos de inferência sintética e perturbações controladas em algoritmos de Inteligência Artificial (IA), com foco em robustez, variabilidade e testes de falhas em datasets e hiperparâmetros.

## 📁 Estrutura do Projeto

```
smart-inference-ai-fusion/
├── main.py                   # Orquestra a execução dos experimentos
├── run.sh                    # Script padrão de execução
├── datasets/                 # Arquivos CSV das bases
├── experiments/              # Subpastas por dataset (ex: iris/, wine/)
│   └── iris/                 # Experimentação com dataset Iris
│       ├── knn_iris.py       # Algoritmo KNN aplicado ao Iris
│       ├── svm_iris.py       # SVM aplicado ao Iris
│       ├── ...
│       └── run_all.py        # Executa todos os experimentos do Iris
├── results/                  # Logs das inferências nos parâmetros
├── src/                      # Código-fonte do framework
│   ├── core/                 # Engine base (Experiment, Model, Dataset)
│   ├── datasets/             # Carregadores padronizados
│   ├── inference/            # Técnicas de inferência (data e params)
│   ├── models/               # Implementações dos algoritmos ML
│   └── utils/                # Métricas e relatórios
└── requirements.txt          # Dependências do projeto
```

## 🚀 Executando

### Ambiente recomendado:
- Python 3.10+
- Ambiente virtual (ex: `venv` ou `conda`)

### Instalação:

```bash
pip install -r requirements.txt
```

### Execução dos experimentos:

```bash
# Forma recomendada
./run.sh

# Ou manualmente:
PYTHONPATH=src python main.py
```

## 🧪 Adicionando novos experimentos

Um experimento corresponde à aplicação de um ou mais algoritmos a um mesmo conjunto de dados (dataset), com e sem inferência.

### Como adicionar um novo experimento:

1. Crie um arquivo em `experiments/<dataset>/`, por exemplo:  
   `experiments/iris/knn_iris.py`
2. Importe o modelo (ex: `KNNModel`) e o dataset loader (ex: `IrisLoader`)
3. Defina duas funções:
   - `run_<algoritmo>_without_inference()`
   - `run_<algoritmo>_with_inference()`
4. Adicione essas chamadas em `experiments/<dataset>/run_all.py`
5. No `main.py`, importe `experiments/<dataset>/run_all` via `__init__.py` da pasta

---

## 🤖 Adicionando novos algoritmos

Os algoritmos são encapsulados em classes e herdados de `BaseModel`.

### Como adicionar um novo algoritmo:

1. Crie um novo arquivo em `src/models/`, ex: `random_forest_model.py`
2. Implemente a classe `RandomForestModel`, com os métodos:
   - `train(self, X_train, y_train)`
   - `evaluate(self, X_test, y_test)` → usando `evaluate_all()`
3. Use esse novo modelo em um experimento como qualquer outro

---

## 🧠 Adicionando novas técnicas de inferência

Técnicas de inferência são classes herdadas de `InferenceTransformation`, aplicadas a dados de entrada.

### Como criar uma nova técnica:

1. Crie ou edite um arquivo em `src/inference/`, ex: `noise.py`
2. Implemente uma nova classe com o método `apply(self, X)`
3. Adicione a lógica correspondente no `InferenceEngine`, dentro de `apply_all()`
4. Atualize o dicionário `config` dos experimentos para ativar a nova técnica

> ⚠️ Técnicas de inferência podem ser aplicadas tanto em **X (dados)** quanto em **y (rótulos)**.

---

## 📂 Adicionando novos datasets

Datasets são carregados por classes que herdam de `BaseDataset`.

### Como adicionar uma nova base:

1. Coloque o arquivo `.csv` em `datasets/<nome>/`, ex: `datasets/wine/wine.csv`
2. Crie um loader em `src/datasets/`, ex: `wine_loader.py`
3. Implemente a classe `WineLoader` com os métodos:
   - `load_X_y(self)` → retorna `X, y`
   - `get_labels(self)` → retorna lista de classes
4. Use o loader em seus experimentos (ex: `experiments/wine/knn_wine.py`)

---

## 🧠 Suporte a Inferência

Este framework suporta inferência em dois níveis:
## 1. Inferência nos dados (data inference)

Técnicas que simulam ruído, falhas e distorções nos dados de entrada.

- Ruído Aditivo: `GaussianNoise, FeatureSelectiveNoise`
- Redução de Precisão: `TruncateDecimals, CastToInt, Quantize`
- Perturbação Estrutural: `ShuffleFeatures, ScaleFeatures, RemoveFeatures, FeatureSwap`
- Corrupção Direta: `ZeroOut, InsertNaN`
- Outliers: `InjectOutliers`
- Distração Semântica: `AddDummyFeatures, DuplicateFeatures`
- Label Noise: `LabelNoise`

## 2. Inferência em parâmetros (parameter inference)
#### O `SmartParameterPerturber` realiza mutações automáticas e validadas em hiperparâmetros dos modelos:
- Alterações baseadas em tipo (`int`, `float`, `str`, `bool`)
- Estratégias como `add_noise`, `cast`, `drop`, `flip,` etc.
- Filtros por nome de parâmetro e validação automática do modelo
- Geração de logs JSON para rastreamento de inferências aplicadas

## 📚 Objetivo
Avaliar a robustez e sensibilidade de algoritmos de IA em cenários realistas com:

- Ruído nos dados
- Falhas de coleta ou sensores
- Valores fora do padrão esperado
- Perturbações nos parâmetros do modelo
---
