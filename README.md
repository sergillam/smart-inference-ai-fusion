# Smart Inference AI Fusion

Um framework modular e extensível para experimentos de inferência sintética e perturbações controladas em algoritmos de Inteligência Artificial (IA), com foco em robustez, variabilidade e testes de falhas em dados e parâmetros.


## 📁 Estrutura do Projeto

```
smart-inference-ai-fusion/
├── main.py                   # Orquestra a execução dos experimentos
├── run.sh                    # Script padrão de execução
├── datasets/                 # Arquivos CSV que representa o dataset
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

Um experimento representa a aplicação de um algoritmo a um dataset, com ou sem inferência aplicada (nos dados e/ou nos parâmetros).


### ✅ 1. Escolha e carregue o **dataset**

Utilize a `DatasetFactory` para criar dinamicamente o carregador com o dataset na origem (`sklearn`, `csv`, etc):

```python
from datasets.factory import DatasetFactory
from utils.types import DatasetSourceType, SklearnDatasetName

dataset = DatasetFactory.create(
    source_type=DatasetSourceType.SKLEARN,
    name=SklearnDatasetName.IRIS
)
```

Para dataset externos (ex: `.csv`), troque o tipo e informe o caminho:
```python
dataset = DatasetFactory.create(
    source_type=DatasetSourceType.CSV,
    path="datasets/wine/wine.csv",
    label_column="target"
)
```

### ✅ 2. Selecione o algoritmo:

Importe o modelo desejado e configure os parâmetros base:

```python
from models.knn_model import KNNModel
base_params = {"n_neighbors": 3, "weights": "uniform"}
model = KNNModel(base_params)
```

### ✅ 3. (Opcional) Adicione inferência nos parâmetros:

Use a função `apply_param_inference` para aplicar perturbações nos hiperparâmetros:
from inference.param_runner import apply_param_inference
```python
model, param_log = apply_param_inference(
    model_class=KNNModel,
    base_params=base_params,
    seed=42,
    ignore_rules={"weights"}  # Evita perturbar esse parâmetro
)
```

### ✅ 4. (Opcional) Adicione inferência nos dados:

Configure e aplique técnicas de perturbação nos dados com o `InferenceEngine`:

from inference.inference_engine import InferenceEngine
```python
config = {
    'noise_level': 0.2,
    'truncate_decimals': 1,
    ...
}
inference = InferenceEngine(config)
```

### ✅ 5. Execute o experimento:

Monte o experimento com ou sem inferência:
```python
from core.experiment import Experiment

experiment = Experiment(model, dataset, inference=inference)
metrics = experiment.run()
```

### ✅ 6. Reporte os resultados:
```python
from utils.report import report_data, ReportMode

report_data(metrics, mode=ReportMode.PRINT)
report_data(param_log, mode=ReportMode.JSON, file_path="results/knn_param_log.json")
```

### ✅ 7. Integre ao sistema de execução:
- Crie uma função run() que chame:
```python
run_<algoritmo>_without_inference()
run_<algoritmo>_with_inference()
```
- Registre essa função em `experiments/<dataset>/run_all.py`

- Inclua no `__init__.py` da pasta `experiments/<dataset>/`

---

## 🤖 Adicionando novos algoritmos

Os algoritmos são encapsulados em classes e herdados de `BaseModel`.

### Como adicionar um novo algoritmo:

1. Crie um novo arquivo em `src/models/`, ex: `knn_model.py`
2. Implemente a classe `KNNModel`, com os métodos:
   - `train(self, X_train, y_train)`
   - `evaluate(self, X_test, y_test)` → usando `evaluate_all()`
3. Use esse novo modelo em um experimento como qualquer outro

---
## 📂 Adicionando novos datasets

O carregamento de datasets é feito através da classe `DatasetFactory`, que seleciona dinamicamente a origem dos dados: `sklearn`, `csv`, ou futuras integrações.

### 💡 Formas de carregar datasets:

- Via ```scikit-learn```: usando o ```SklearnDatasetLoader```

- Via `.csv`: usando o `CSVDatasetLoader`

### 🛠️ Como adicionar um novo dataset:
### 1. Se a origem for sklearn:
- Registre o nome do dataset no enum `SklearnDatasetName`:
    ```python
    class SklearnDatasetName(Enum):
        IRIS = "iris"
        WINE = "wine"
    ```
- Use no experimento:
    ```python
    dataset = DatasetFactory.create(DatasetSourceType.SKLEARN, name=SklearnDatasetName.IRIS)
    ```

### 2. Se a origem for um arquivo CSV:

- Coloque o arquivo `.csv` em `datasets/<nome>/`, ex: `datasets/iris/iris.csv`

- Use o `CSVDatasetLoader` no experimento:
    ```python
    dataset = DatasetFactory.create(
        DatasetSourceType.CSV,
        path="datasets/wine/wine.csv",
        label_column="target"
    )
    ```

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


## 🧩 Enumerações e Tipagens do Framework
Essa seção serve como referência rápida para desenvolvedores que forem integrar novos módulos, criar loaders ou escrever experimentos

Local: `src/utils/types.py`
```python
class DatasetSourceType(Enum):
    SKLEARN = "sklearn"
    CSV = "csv"

class ReportMode(Enum):
    PRINT = "print"
    JSON = "json"

class SklearnDatasetName(Enum):
    IRIS = "iris"
    WINE = "wine"
```
## 📚 Objetivo
Avaliar a robustez e sensibilidade de algoritmos de IA em cenários realistas com:

- Ruído nos dados
- Falhas de coleta ou sensores
- Valores fora do padrão esperado
- Perturbações nos parâmetros do modelo
---
