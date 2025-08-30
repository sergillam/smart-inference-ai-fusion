# Smart Inference AI Fusion

Um framework modular e extensível para experimentos de inferência sintética e perturbações controladas em algoritmos de Inteligência Artificial (IA), com foco em robustez, variabilidade e testes de falhas em dados e parâmetros.


## 📁 Estrutura do Projeto
```
├── pyproject.toml               # Arquivo de configuração que gerencia dependências e build do projeto
├── makefile                     # Comandos automatizados para execução, lint, testes etc.
├── README.md                    # Documentação principal do projeto
├── datasets/                    # Contém os datasets não oriundos do scikit-learn (ex: arquivos .csv)
├── docs/                        # Documentação adicional e resumos
├── logs/                        # Logs de execução e inferência
├── results/                     # Resultados dos experimentos e inferências
├── scripts/                     # Contém scripts utilizados para automação de configuração;
├── smart_inference_ai_fusion/   # Código-fonte principal do framework
│   ├── core/                    # Classes base para Experimento, Modelo e Dataset
│   ├── datasets/                # Módulos para carregar datasets (de arquivos CSV, scikit-learn etc.)
│   ├── experiments/             # Scripts de experimentos, agora parte do pacote, com um `run.py` orquestrador
│   ├── inference/               # Módulo central de inferência de ruídos
│   │   ├── engine/              # Motores que orquestram a aplicação das perturbações
│   │   ├── pipeline/            # Pipeline que integra e aplica as transformações
│   │   └── transformations/     # Lógicas de perturbação, separadas por alvo:
│   │       ├── data/            # Técnicas aplicadas aos dados de entrada (X)
│   │       ├── label/           # Técnicas aplicadas aos rótulos (y)
│   │       └── params/          # Estratégias de perturbação nos hiperparâmetros
│   ├── models/                  # Wrappers de modelos (BaseModel-compatíveis)
│   └── utils/                   # Funções utilitárias, tipos, métricas e relatórios
└── tests/                       # Testes unitários do framework
```

## 🚀 Guia de Instalação e Execução
### Este guia assume que você está em um ambiente `Linux` ou `MacOS`.
#### **Pré-requisitos**:
- `Git`
- `Python 3.10`
- `Make` para usar os comandos automatizados

#### **Instalação (Método Recomendado: `make`)**

O `Makefile` automatiza todo o processo de configuração. Você só precisa de um comando.
1. **Clone o repositório:**
    ```bash
    git clone git@github.com:sergillam/smart-inference-ai-fusion.git
    cd smart-inference-ai-fusion
    ```

2. **Instale as dependências:**
    
    - **Para desenvolver no projeto (Recomendado):**
    
        Este comando cria o ambiente virtual, instala todas as dependências (de execução, testes, linting, etc.) e o pacote em modo editável.
        ```bash
        make install-dev
        ```
    
    - **Apenas para executar os experimentos:**
    

        Este comando faz uma instalação mínima, apenas com as dependências de execução.
        ```bash
        make install
        ```

    - **Nota: Você não precisa criar ou ativar o ambiente virtual (venv) manualmente. Os comandos make cuidam de tudo para você.**

#### **Executando os Experimentos**

Use `make run` para executar os scripts. A variável `EXP` define o alvo.

1. **Executar Todos os Experimentos**
    
    Roda o orquestrador principal que executa todos os scripts de experimento.
    ```bash
    make run
    ```

2. **Executar um Pacote de Experimentos**
    
    Roda todos os experimentos de um diretório específico.
    ```bash
    make run EXP=smart_inference_ai_fusion.experiments.name_data_set
    ```

3. **Executar um Único Experimentos**
    
    Roda um único arquivo de experimento.
    ```bash
    make run EXP=smart_inference_ai_fusion.experiments.name_data_set.name_experiments
    ```

4. **Passando Argumentos para os Scripts**
    
    Use a variável `ARGS` para passar argumentos customizados para seus scripts.
    ```bash
    make run EXP=<seu_alvo> ARGS="--seed 42 --outro_parametro valor"
    ```

5. **Executar em Modo Debug**
    
    Para uma saída de log mais detalhada.
    ```bash
    make debug
    ```

#### **Workflow de Desenvolvimento**

O `Makefile` inclui vários comandos para garantir a qualidade e a manutenção do código.

- `make check`: Roda todas as verificações de qualidade (formatação, linting e estilo de docstrings).
- `make format`: Formata o código automaticamente.
- `make test`: Roda a suíte de testes unitários.
- `make clean-outputs`: Limpa as pastas de `logs` e `results` (útil para garantir uma execução limpa).
- `make help`: Lista todos os comandos disponíveis e o que eles fazem.

#### **Instalação Manual (Alternativa)**
Para usuários de `Windows` ou que não desejam usar `make`.
```bash
# 1. Clone o repositório e entre na pasta

git clone git@github.com:sergillam/smart-inference-ai-fusion.git
cd smart-inference-ai_fusion

# 2. Crie e ative o ambiente virtual
# No Windows:
python -m venv .venv
.\.venv\Scripts\activate

# No Linux / MacOS:
# python3.10 -m venv .venv
# source .venv/bin/activate

# 3. Instale o projeto e suas dependências
# O comando abaixo lê o `pyproject.toml` e instala tudo
pip install -e .[dev]

# Para instalação mínima (sem ferramentas de dev), use:
# pip install -e .
```

## 🧪 Adicionando novos experimentos

Um experimento representa a aplicação de um algoritmo a um dataset, com ou sem inferência aplicada (nos dados e/ou nos parâmetros).


### ✅ 1. Escolha e carregue o **dataset**

Utilize a `DatasetFactory` para criar dinamicamente o carregador com o dataset na origem (`sklearn`, `csv`, etc):

```python
from datasets.factory import DatasetFactory
from utils.types import DatasetSourceType, SklearnDatasetName

# Exemplo: carregar o dataset Iris do sklearn
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
Como obter os splits:

Obtenha os dados já separados em treino/teste (80% treino, 20% teste)
```python
X_train, X_test, y_train, y_test = dataset.load_split(test_size=0.2)
```
💡 Dica: Consulte a docstring da `DatasetFactory` para outros tipos suportados, como outros datasets do sklearn, arquivos csv customizados, etc.

### ✅ 2. Defina o modelo (algoritmo de IA):

Importe o modelo desejado e configure os parâmetros base:

```python
from models.knn_model import KNNModel
base_params = {"n_neighbors": 3, "weights": "uniform"}
model = KNNModel(base_params)
```
ℹ️ Observação: O framework possui modelos prontos para KNN, SVM, Decision Tree, Perceptron, GaussianNB e outros. Consulte o diretório `src/models/` para ver todos os disponíveis.

➡️ Próximo passo: Configure sua pipeline de inferência para aplicar perturbações nos dados, rótulos ou parâmetros do modelo (veja o Passo 3).

###  ✅ 3. Adicione inferência com o `InferencePipeline`

#### ✅ 3.1 Inferência nos parâmetros
- Informe as configurações separadas para dados `DataNoiseConfig`, rótulos `LabelNoiseConfig` e parâmetros do modelo `ParameterNoiseConfig`.
Essas configs são validadas por Pydantic e cada uma ativa diferentes técnicas:

    ```python
    from utils.types import DataNoiseConfig, LabelNoiseConfig, ParameterNoiseConfig

    data_noise_config = DataNoiseConfig(
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
        conditional_noise=(0, 5.0, 0.2),
        random_missing_block_fraction=0.1,
        distribution_shift_fraction=0.1,
        cluster_swap_fraction=0.1,
        group_outlier_cluster_fraction=0.1,
        temporal_drift_std=0.5,
    )

    label_noise_config = LabelNoiseConfig(
        label_noise_fraction=0.1,
        flip_near_border_fraction=0.1,
        confusion_matrix_noise_level=0.1,
        partial_label_fraction=0.1,
        swap_within_class_fraction=0.1,
    )

    param_noise_config = ParameterNoiseConfig(
        integer_noise=True,
        boolean_flip=True,
        string_mutator=True,
        semantic_mutation=True,
        scale_hyper=True,
        cross_dependency=True,
        random_from_space=True,
        bounded_numeric=True,
        type_cast_perturbation=True,
        enum_boundary_shift=True,
    )
    ```

#### ✅ 3.2 Instancie a pipeline de inferência
- Instancie o pipeline passando as configs desejadas (todas opcionais):
    ```python
    from inference.pipeline.inference_pipeline import InferencePipeline

    pipeline = InferencePipeline(
        data_noise_config=data_noise_config,
        label_noise_config=label_noise_config,
        X_train=X_train,  # X_train é necessário para algumas técnicas de rótulo
    )
    ```

#### ✅ 3.3 Aplique as perturbações
- Dados (X):
    ```python
    X_train_pert, X_test_pert = pipeline.apply_data_inference(X_train, X_test)
    ```
- Rótulos (y):
    ```python
    y_train_pert, y_test_pert = pipeline.apply_label_inference(
        y_train, y_test,
        model=model,             # obrigatório para algumas técnicas (ex: flip_near_border)
        X_train=X_train_pert,    # passe os dados já perturbados, se aplicável
        X_test=X_test_pert
    )
    ```
- Parâmetros (opcional, para perturbar hiperparâmetros):
    ```python
    from models.gaussian_model import GaussianNBModel

    model, param_log = pipeline.apply_param_inference(
        model_class=GaussianNBModel,
        base_params={"var_smoothing": 1e-9}
    )
    ```
##### Obs:
- Passe apenas as configs de interesse. Se não quiser inferência em algum aspecto, omita o argumento (por exemplo, `InferencePipeline(data_noise_config=data_noise_config))`.

- Consulte os exemplos em `experiments/` para fluxos completos.

### ✅ 4. Execute o experimento:

- Monte o experimento com ou sem inferência:
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
- Crie funções públicas de entrada para seus experimentos, seguindo o padrão abaixo:
    ```python
    # experiments/<dataset>/<seu_experimento>.py

    def run_<algoritmo>_without_inference():
        # Configuração e execução SEM inferência
        pass

    def run_<algoritmo>_with_inference():
        # Configuração e execução COM inferência
        pass

    def run():
        """Executa todas as variantes deste experimento."""
        run_<algoritmo>_without_inference()
        run_<algoritmo>_with_inference()
    ```
- No arquivo `experiments/<dataset>/__init__.py`, importe e chame os experimentos desse dataset:
    ```python
    # experiments/<dataset>/__init__.py

    from .<seu_experimento> import run as run_<algoritmo>

    def run_all():
        print("=== Executando todos os experimentos para <dataset> ===")
        run_<algoritmo>()  # Adicione mais funções se tiver outros experimentos
        print("=== Experimentos concluídos ===")
    ```
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
Cada técnica de inferência é representada por uma classe que herda de `InferenceTransformation`, `LabelTransformation` ou `ParameterTransformation`, e implementa o método `apply(...)`.

### ✅ 1. Técnicas aplicadas aos dados (X)

Passos:
1. Crie um novo arquivo ou edite um existente em `src/inference/transformations/data/`.

2. Implemente a nova classe herdando de `InferenceTransformation` e o método `apply(self, X)`.
    ```python
    class MinhaTransformacao(InferenceTransformation):
        def apply(self, X):
            # Sua lógica aqui
            return X_modificado
    ```
3. Registre a nova transformação no pipeline em `src/inference/engine/inference_engine.py`, adicionando uma verificação no construtor e incluindo no pipeline.

4. Adicione um campo correspondente (se desejar configurável) no `DataNoiseConfig` (`src/utils/types.py`).

5. Utilize nos experimentos via `InferencePipeline`.

### ✅ 2. Técnicas aplicadas aos rótulos (y)

Passos:
1. Crie uma nova classe em `src/inference/transformations/label/`, herdando de `LabelTransformation` (ou base similar).

2. Implemente o método `apply(self, y)`  (ou `apply(self, y, X=None, model=None`) se precisar).

3. Registre a técnica no pipeline em (`src/inference/engine/label_runner.py`).

4. Configure no campo adequado em `LabelNoiseConfig` (em `src/utils/types.py`).

### ✅ 3. Técnicas aplicadas aos parâmetros dos modelos

Essas técnicas são tratadas por `SmartParameterPerturber`.

1. Crie uma nova classe de transformação em `src/inference/transformations/params/` herdando de `ParameterTransformation` e implementando o método `apply(self, params)`.

2. Importe e registre a nova técnica no pipeline em `src/inference/engine/param_runner.py` (na lista de técnicas disponíveis).

3. Se necessário, adicione uma flag de configuração em `ParameterNoiseConfig` (`src/utils/types.py`).

4. O ponto de entrada para orquestração é o método `apply_param_inference()` da pipeline.

#### ⚠️ As três categorias são independentes, mas integradas por meio da `InferencePipeline`. Você pode aplicar apenas uma, duas ou todas combinadas nos seus experimentos.
---

## 🧬 Suporte a Inferência
Este framework suporta inferência sintética em múltiplos níveis, possibilitando testes de robustez em diferentes etapas do pipeline de IA:
## 1. Inferência nos dados (data inference)

#### Técnicas para simular ruído, falhas e distorções nos dados de entrada (`X`) :

- **Ruído Aditivo**: `GaussianNoise, FeatureSelectiveNoise`
- **Redução de Precisão**: `TruncateDecimals, CastToInt, Quantize`
- **Perturbação Estrutural**: `ShuffleFeatures, ScaleFeatures, RemoveFeatures, FeatureSwap`
- **Corrupção Direta**: `ZeroOut, InsertNaN`
- **Outliers**: `InjectOutliers`
- **Distração Semântica**: `AddDummyFeatures, DuplicateFeatures`
- **Perturbações tabulares avançadas**: `RandomMissingBlock`, `DistributionShiftMixing`, `ClusterSwap`, `GroupOutlierInjection`, `TemporalDriftInjection`.

## 2. Inferência nos rótulos (label inference)
#### Técnicas para simular erros e ambiguidade nos rótulos (`y`):
- **Ruído aleatório**: `RandomLabelNoise`
- **Ruído guiado por matriz de confusão:** `LabelConfusionMatrixNoise`
- **Flip near border** (baixa confiança do modelo): `LabelFlipNearBorder`
- **Ambiguidade parcial**: `PartialLabelNoise`
- **Troca de rótulos dentro de classes**: `LabelSwapWithinClass`

## 3. Inferência em parâmetros (parameter inference)
#### O `SmartParameterPerturber` realiza mutações automáticas e validadas em hiperparâmetros dos modelos, incluindo:
- **Alterações baseadas em tipo**: (`int`, `float`, `str`, `bool`)
- **Estratégias**: `add_noise`, `cast`, `drop`, `flip`, entre outras.
- **Filtros por nome de parâmetro e validação automática do modelo**
- **Geração de logs JSON para rastreamento das inferências aplicadas**.

#### ⚡ Nota: As três categorias são independentes, mas integradas via a `InferencePipeline`. Você pode aplicar apenas uma, duas ou todas as inferências combinadas em seus experimentos.

## 🧩 Enumerações e Tipagens do Framework
As definições que padronizam origem de dados, nomes de datasets, modos de relatório e configs de perturbação ficam em  `src/utils/types.py`

- **DatasetSourceType** → origem dos dados: `SKLEARN`, `CSV`.
- **SklearnDatasetName** → conjuntos suportados: `IRIS`, `WINE`, `BREAST_CANCER`, `DIGITS`.
- **CSVDatasetName** → atalhos de CSV (ex.: `TITANIC`) com propriedade `.path`.
- **ReportMode** → destino do output: `PRINT` (console), `JSON_LOG` (`logs/`), `JSON_RESULT` (`results/`).

### 🔧 Configurações de Perturbação (Pydantic)

O framework utiliza **Pydantic** para gerenciar e validar as configurações de perturbação em **dados (X)**, **rótulos (y)** e **hiperparâmetros**.  
As configurações são definidas pelas classes `DataNoiseConfig`, `LabelNoiseConfig` e `ParameterNoiseConfig`.

---

#### 📊 `DataNoiseConfig` – Perturbações nos Dados de Entrada (X)

Controla ruídos, transformações estruturais e distorções tabulares para testar robustez dos modelos.

| Categoria | Parâmetro | Tipo / Exemplo | Descrição |
|-----------|-----------|---------------|-----------|
| **Ruído e Precisão** | `noise_level` | `0.1` | Intensidade de ruído gaussiano. |
| | `truncate_decimals` | `2` | Trunca valores para N casas decimais. |
| | `quantize_bins` | `5` | Quantização dos dados em N bins. |
| | `cast_to_int` | `True` | Converte valores para inteiros. |
| **Estrutura e Escala** | `shuffle_fraction` | `0.1` | Fração de colunas embaralhadas. |
| | `scale_range` | `(0.8, 1.2)` | Escala de valores (min, max). |
| | `remove_features` | `[1, 3]` | Índices de features a remover. |
| | `feature_swap` | `[0, 2]` | Índices de features a trocar entre si. |
| **Corrupção e Outliers** | `zero_out_fraction` | `0.05` | Fração de valores zerados. |
| | `insert_nan_fraction` | `0.05` | Fração de `NaN`s inseridos. |
| | `outlier_fraction` | `0.05` | Fração de outliers aleatórios. |
| **Distrações e Redundância** | `add_dummy_features` | `2` | Número de features fictícias adicionadas. |
| | `duplicate_features` | `2` | Número de features duplicadas. |
| **Perturbações Avançadas** | `feature_selective_noise` | `(0.3, [0, 2])` | Aplica ruído específico em features selecionadas. |
| | `conditional_noise` | `(0, 5.0, 0.2)` | Ruído condicional (feature, valor, desvio). |
| | `random_missing_block_fraction` | `0.1` | Porção de blocos inteiros de dados ausentes. |
| | `distribution_shift_fraction` | `0.1` | Mudança de distribuição simulada. |
| | `cluster_swap_fraction` | `0.1` | Troca de amostras entre clusters. |
| | `group_outlier_cluster_fraction` | `0.1` | Introdução de grupos de outliers. |
| | `temporal_drift_std` | `0.5` | Desvio padrão do drift temporal. |

---

#### 🏷️ `LabelNoiseConfig` – Perturbações em Rótulos (y)

Aplica ruídos e distorções controladas nos rótulos para simular erros de anotação.

| Parâmetro | Tipo / Exemplo | Descrição |
|-----------|---------------|-----------|
| `label_noise_fraction` | `0.05` | Fração de rótulos aleatoriamente alterados. |
| `flip_near_border_fraction` | `0.05` | Troca rótulos próximos da fronteira de decisão. |
| `confusion_matrix_noise_level` | `0.05` | Probabilidade de ruído guiado por matriz de confusão. |
| `partial_label_fraction` | `0.05` | Fração de rótulos substituídos por conjuntos parciais. |
| `swap_within_class_fraction` | `0.05` | Troca de rótulos dentro da mesma classe. |

---

#### ⚙️ `ParameterNoiseConfig` – Perturbações em Hiperparâmetros

Simula cenários adversos ao modificar hiperparâmetros de modelos.

| Parâmetro | Tipo / Exemplo | Descrição |
|-----------|---------------|-----------|
| `integer_noise` | `True` | Aplica ruído em hiperparâmetros inteiros. |
| `boolean_flip` | `False` | Inverte valores booleanos. |
| `string_mutator` | `False` | Altera strings de parâmetros (ex: nomes de otimizadores). |
| `semantic_mutation` | `False` | Perturba valores respeitando semântica (ex: step size). |
| `scale_hyper` | `True` | Escala valores numéricos (multiplicativo). |
| `cross_dependency` | `False` | Perturba parâmetros considerando dependências cruzadas. |
| `random_from_space` | `False` | Escolhe valores aleatórios de espaços pré-definidos. |
| `bounded_numeric` | `True` | Garante que valores numéricos fiquem em faixas válidas. |
| `type_cast_perturbation` | `False` | Converte tipos dinamicamente (int ↔ float). |
| `enum_boundary_shift` | `False` | Escolhe próximo valor válido em enums. |

---

## 📚 Objetivo

Avaliar a robustez e a sensibilidade de modelos de IA sob **perturbações controladas** em dados, rótulos e hiperparâmetros, simulando:

- ruído e perdas de precisão;  
- falhas de coleta/sensores e valores ausentes;  
- outliers e mudanças de distribuição;  
- erros de anotação e variações de configuração do modelo.  

---

## 📄 Publicação

**Sergillam Barroso Oliveira**, **Eddie B de Lima Filho**, **Lucas Cordeiro**.  
*Modular Architecture for Robustness Assessment in AI Using Smart Inference*.  
In: **2025 IEEE 14th Global Conference on Consumer Electronics (GCCE)** — Track OS-DSC, apresentação oral (*Student Paper*).  
EDAS ID: **1571156254**.  

### 📌 Como citar
```bibtex
@inproceedings{oliveira2025modular,
  author    = {Sergillam Barroso Oliveira and Eddie B de Lima Filho and Lucas Cordeiro},
  title     = {Modular Architecture for Robustness Assessment in AI Using Smart Inference},
  booktitle = {Proceedings of the 2025 IEEE 14th Global Conference on Consumer Electronics (GCCE)},
  year      = {2025},
  month     = sep,
  publisher = {IEEE},
  note      = {Track: OS-DSC (Oral, Student Paper), EDAS ID: 1571156254}
}
```
---
