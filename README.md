# Smart Inference AI Fusion

Um framework modular e extensível para experimentos de inferência sintética e perturbações controladas em algoritmos de Inteligência Artificial (IA), com foco em robustez, variabilidade e testes de falhas em dados e parâmetros.


## 📁 Estrutura do Projeto

```
smart-inference-ai-fusion/
├── main.py                      # Ponto de entrada principal para execução dos experiments
├── run.sh                       # Script utilitário para execução rápida
├── datasets/                    # Arquivos CSV utilizados por loaders do tipo CSVDatasetLoader
├── experiments/                 # Experimentos organizados por dataset (ex: iris/, wine/)
│   └── iris/
│       ├── knn_iris.py          # KNN aplicado ao dataset Iris
│       ├── svm_iris.py          # SVM aplicado ao dataset Iris
│       ├── run_all.py           # Executa todos os experimentos do Iris
│       └── ...
├── results/                     # Resultados e logs de inferência (ex: parâmetros perturbados)
├── src/                         # Código-fonte principal do framework
│   ├── core/                    # Classes base para Experiment, Model e Dataset
│   ├── datasets/                # Factory e loaders de datasets (sklearn, csv, etc.)
│   ├── inference/               # Módulo de inferência
│   │   ├── engine/              # Orquestradores de inferência (InferenceEngine, LabelRunner, etc.)
│   │   ├── pipeline/            # Pipeline unificada que aplica todas as inferências
│   │   ├── transformations/
│   │   │   ├── data/            # Técnicas aplicadas aos dados de entrada (X)
│   │   │   ├── label/           # Técnicas aplicadas aos rótulos (y)
│   │   │   └── params/          # Estratégias de perturbação nos parâmetros
│   ├── models/                  # Implementações dos modelos de IA (KNN, SVM, Tree, etc.)
│   ├── utils/                   # Relatórios, enums, tipos e métricas
│   │   └── types.py             # Tipos pydantic e enums como DatasetSourceType, ReportMode
└── requirements.txt             # Lista de dependências do projeto

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

### ✅ 2. Defina o modelo (algoritmo de IA):

Importe o modelo desejado e configure os parâmetros base:

```python
from models.knn_model import KNNModel
base_params = {"n_neighbors": 3, "weights": "uniform"}
model = KNNModel(base_params)
```

###  ✅ 3. Adicione inferência com o `InferencePipeline`

#### ✅ 3.1 Inferência nos parâmetros
Use a função `apply_param_inference` para aplicar perturbações nos hiperparâmetros:
```python
from inference.pipeline.inference_pipeline import InferencePipeline

pipeline = InferencePipeline()

model, param_log = pipeline.apply_param_inference(
    model_class=KNNModel,
    base_params=base_params,
    seed=42,
    ignore_rules={"weights"}
)
```

#### ✅ 3.2 Inferência nos dados (X) e nos rótulos (y)
- Crie o dicionário com as configurações de perturbação validadas por Pydantic:
```python
from utils.types import DatasetNoiseConfig

dataset_noise_config = DatasetNoiseConfig(
    noise_level=0.2,
    truncate_decimals=1,
    quantize_bins=5,
    cast_to_int=False,
    shuffle_fraction=0.1,
    scale_range=(0.8, 1.2),
    zero_out_fraction=0.05,
    insert_nan_fraction=0.05,
    outlier_fraction=0.05,
    add_dummy_features=2,
    duplicate_features=2,
    feature_selective_noise=(0.3, [0, 2]),
    remove_features=[1, 3],
    feature_swap=[0, 2],
    label_noise_fraction=0.1
)
```

#### ✅ 3.3 Aplique a inferência com `InferencePipeline`
```python
pipeline = InferencePipeline(dataset_noise_config=dataset_noise_config)

X_train, X_test, y_train, y_test = dataset.load_data()

X_train, X_test = pipeline.apply_data_inference(X_train, X_test)
y_train, y_test = pipeline.apply_label_inference(y_train, y_test)
```

### ✅ 4. Execute o experimento:

Monte o experimento com ou sem inferência:
```python
from core.experiment import Experiment

experiment = Experiment(model, dataset)
metrics = experiment.run(X_train, X_test, y_train, y_test)
```

### ✅ 5. Reporte os resultados:
```python
from utils.report import report_data, ReportMode

report_data(metrics, mode=ReportMode.PRINT)
report_data(param_log, mode=ReportMode.JSON, file_path="results/knn_param_log.json")
```

### ✅ 6. Integre ao sistema de execução:
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

Técnicas de inferência são componentes modulares aplicadas aos dados de entrada (X), aos rótulos (y) ou aos parâmetros (kwargs) dos modelos.

### Estrutura geral
Cada técnica de inferência é representada por uma classe que herda de `InferenceTransformation` (ou equivalente) e implementa o método `apply(...)`.

### ✅ 1. Técnicas aplicadas aos dados (X)
Cada técnica de inferência é representada por uma classe que herda de `InferenceTransformation` (ou equivalente) e implementa o método `apply(...)`.

Passos:
1. Crie um novo arquivo ou edite um existente em `src/inference/transformations/data/`.

2. Crie uma nova classe com o método:
```python
class MinhaTransformacao(InferenceTransformation):
    def apply(self, X):
        # sua lógica de transformação
        return X_modificado
```
3. Registre a nova transformação em `src/inference/engine/inference_engine.py`, adicionando uma verificação no construtor e incluindo no pipeline.

4. Adicione um campo correspondente no `DatasetNoiseConfig` (com validação Pydantic).

5. Utilize nos experimentos via `InferencePipeline`.

### ✅ 2. Técnicas aplicadas aos rótulos (y)

Passos:
1. Crie uma nova classe em `src/inference/transformations/label/`, herdando de LabelTransformation (ou base similar).

2. Implemente o método `apply(self, y)`.

3. Registre no `LabelInferenceEngine` (`src/inference/engine/label_runner.py`).

4. Configure no campo `label_noise_fraction` ou crie um novo campo em `DatasetNoiseConfig`.

### ✅ 3. Técnicas aplicadas aos parâmetros dos modelos

Essas técnicas são tratadas por `SmartParameterPerturber`.

1. A lógica de perturbação está em `src/inference/transformations/params/parameter_perturber.py`.

2. Para novas estratégias (ex: `replace_with_random`), adicione métodos internos na classe.

3. O ponto de entrada principal está em `src/inference/engine/param_runner.py` via a função `apply_param_inference()`.

#### ⚠️ As três categorias são independentes, mas integradas por meio da InferencePipeline. Você pode aplicar apenas uma, duas ou todas combinadas.
---

## 🧰 Suporte a Inferência

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
    
class DatasetNoiseConfig(BaseModel):
    noise_level: Optional[float] = None  # Intensidade de ruído gaussiano
    truncate_decimals: Optional[int] = None  # Número de casas decimais
    quantize_bins: Optional[int] = None  # Quantização em N bins
    cast_to_int: Optional[bool] = None  # Converte para int
    shuffle_fraction: Optional[float] = None  # Fração de colunas embaralhadas
    scale_range: Optional[Tuple[float, float]] = None  # Intervalo de escala (min, max)
    zero_out_fraction: Optional[float] = None  # Fração de valores zerados
    insert_nan_fraction: Optional[float] = None  # Fração de NaNs inseridos
    outlier_fraction: Optional[float] = None  # Fração de outliers
    add_dummy_features: Optional[int] = None  # N novas features aleatórias
    duplicate_features: Optional[int] = None  # N features duplicadas
    feature_selective_noise: Optional[Tuple[float, List[int]]] = None  # (nível, índices)
    remove_features: Optional[List[int]] = None  # Índices a remover
    feature_swap: Optional[List[int]] = None  # Índices a trocar entre si
    label_noise_fraction: Optional[float] = None  # Ruído nos rótulos
```
## 📚 Objetivo
Avaliar a robustez e sensibilidade de algoritmos de IA em cenários realistas com:

- Ruído nos dados
- Falhas de coleta ou sensores
- Valores fora do padrão esperado
- Perturbações nos parâmetros do modelo
---
