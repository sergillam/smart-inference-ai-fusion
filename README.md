# Smart Inference AI Fusion

Um framework modular e extensível para experimentos de inferência sintética e perturbações controladas em algoritmos de Inteligência Artificial (IA), com foco em **verificação formal multi-solver** aplicando variabilidade e testes de falhas em dados, labels e parâmetros de modelos.

## 🎯 Características Principais

- **🔬 Verificação Formal Multi-Solver**: Integração com Z3 e CVC5 para validação matemática de propriedades
- **🧪 Framework Modular**: Arquitetura plugável para experimentos reproduzíveis
- **⚡ Pipeline de Inferência**: Perturbações controladas em dados, labels e parâmetros
- **📈 Comparação Sistemática**: Framework automático de comparação entre solvers SMT
- **🔄 Reprodutibilidade**: Configuração versionada e logging estruturado

## 🔧 Arquitetura Multi-Solver

### Sistema de Plugins de Verificadores
```python
# Plugins disponíveis
smart_inference_ai_fusion/verification/plugins/
├── z3_plugin.py          # Plugin Z3 SMT Solver
├── cvc5_plugin.py        # Plugin CVC5 SMT Solver  
└── base_plugin.py        # Interface base para novos solvers
```

### Constraints Implementados
- **Integridade de Dados**: Preservação de forma, tipo e bounds
- **Integridade de Labels**: Validação de tipos e ranges
- **Integridade de Parâmetros**: Verificação de pré/pós-perturbação
- **Aritmética Real/Inteira**: Constraints matemáticos avançados

## 📁 Estrutura do Projeto
```
├── pyproject.toml               # Arquivo de configuração que gerencia dependências e build do projeto
├── makefile                     # Comandos automatizados para execução, lint, testes etc.
├── README.md                    # Documentação principal do projeto
├── configs/                     # Configurações de experimentos avançados
├── datasets/                    # Contém os datasets não oriundos do scikit-learn (ex: arquivos .csv)
├── docs/                        # Documentação adicional e resumos
├── examples/                    # Exemplos de uso do framework de verificação
├── logs/                        # Logs de execução e verificação formal
├── results/                     # Resultados dos experimentos e verificações SMT
├── scripts/                     # Scripts de automação e configuração
├── smart_inference_ai_fusion/   # Código-fonte principal do framework
│   ├── core/                    # Classes base para Experimento, Modelo e Dataset
│   ├── datasets/                # Módulos para carregar datasets (CSV, scikit-learn etc.)
│   ├── experiments/             # Scripts de experimentos organizados por dataset:
│   │   ├── wine/                # Experimentos dataset Wine (logistic, tree, mlp)
│   │   ├── adult/               # Experimentos dataset Adult (logistic, tree, mlp) 
│   │   ├── breast_cancer/       # Experimentos dataset Breast Cancer (logistic, tree, mlp)
│   │   ├── make_moons/          # Experimentos dataset Make Moons (logistic, tree, mlp)
│   │   └── experiment_registry.py # Registry central de experimentos
│   ├── inference/               # Módulo central de inferência de ruídos
│   │   ├── engine/              # Motores que orquestram a aplicação das perturbações
│   │   ├── pipeline/            # Pipeline que integra e aplica as transformações
│   │   └── transformations/     # Lógicas de perturbação, separadas por alvo:
│   │       ├── data/            # Técnicas aplicadas aos dados de entrada (X)
│   │       ├── label/           # Técnicas aplicadas aos rótulos (y)
│   │       └── params/          # Estratégias de perturbação nos hiperparâmetros
│   ├── models/                  # Wrappers de modelos (BaseModel-compatíveis)
│   ├── verification/            # Sistema de verificação formal multi-solver
│   │   ├── plugins/             # Plugins para diferentes solvers SMT
│   │   │   ├── z3_plugin.py     # Plugin Z3 SMT Solver
│   │   │   ├── cvc5_plugin.py   # Plugin CVC5 SMT Solver
│   │   │   └── base_plugin.py   # Interface base para novos solvers
│   │   ├── framework/           # Framework de comparação multi-solver
│   │   └── constraints/         # Definições de constraints formais
│   └── utils/                   # Funções utilitárias, tipos, métricas e relatórios
└── tests/                       # Testes unitários do framework
```

### Características dos Modelos por Dataset
- **LogisticRegression**: Otimizado para convergência rápida com solver `liblinear`
- **DecisionTree**: Profundidade controlada, critério `gini` para classificação
- **MLPModel**: Arquitetura adaptativa com early stopping e regularização

## 🚀 Guia de Instalação e Execução

### Pré-requisitos
- **Git** 
- **Python 3.10+**
- **Make** para comandos automatizados
- **Sistema Operacional**: Linux ou MacOS

### Instalação Rápida

O `Makefile` automatiza todo o processo de configuração:

```bash
# 1. Clone o repositório
git clone git@github.com:sergillam/smart-inference-ai-fusion.git
cd smart-inference-ai-fusion

# 2. Instalação completa para desenvolvimento (recomendado)
make install-dev

# OU instalação mínima apenas para execução
make install
```

**Nota**: Não é necessário criar ou ativar ambiente virtual manualmente. Os comandos `make` gerenciam tudo automaticamente.

## 🧪 Execução de Experimentos

### Comandos Básicos

#### 1. **Verificação Formal com Multi-Solver**
```bash
# Executar dataset específico com Z3
make run verify EXP=wine SOLVERS=z3

# Executar com ambos os solvers (Z3 + CVC5)  
make run verify EXP=adult SOLVERS="z3,cvc5"

# Executar todos os datasets com comparação multi-solver
make run verify SOLVERS="z3,cvc5"
```

#### 2. **Experimentos por Dataset**
```bash
# Dataset Wine (3 experimentos: logistic, tree, mlp)
make run verify EXP=wine SOLVERS=z3

# Dataset Adult (com preprocessamento categórico)
make run verify EXP=adult SOLVERS=cvc5  

# Dataset Breast Cancer (classificação binária)
make run verify EXP=breast_cancer SOLVERS="z3,cvc5"

# Dataset Make Moons (sintético 2D)
make run verify EXP=make_moons SOLVERS=z3
```

#### 3. **Experimentos Específicos**
```bash
# Executar apenas modelo específico
make run verify EXP=wine.logistic SOLVERS=z3
make run verify EXP=adult.tree SOLVERS=cvc5
make run verify EXP=make_moons.mlp SOLVERS="z3,cvc5"
```

### Configuração Avançada

#### Variáveis de Ambiente
```bash
# Controle de logging detalhado
LOG_LEVEL=DEBUG make run verify EXP=wine SOLVERS=z3

# Configuração de timeout para verificação
VERIFICATION_TIMEOUT=60.0 make run verify EXP=adult SOLVERS=cvc5

# Modo de verificação (strict/flexible) 
VERIFICATION_MODE=strict make run verify EXP=breast_cancer SOLVERS="z3,cvc5"
```

## 🔬 Verificação Formal Multi-Solver

### Sistema de Constraints Implementados

O framework implementa verificação formal usando SMT solvers (Z3 e CVC5) para validar propriedades matemáticas:

#### **Constraints de Integridade de Dados**
- **shape_preservation**: Preservação da dimensionalidade dos dados
- **type_safety**: Validação de tipos (float64, int32, etc.)
- **bounds**: Verificação de limites mínimos e máximos
- **range_check**: Validação de ranges válidos contínuos

#### **Constraints de Integridade de Labels**  
- **shape_preservation**: Preservação do número de labels
- **type_safety**: Validação de tipos de labels
- **bounds**: Verificação de bounds dos labels
- **integer_arithmetic**: Validação de aritmética inteira

#### **Constraints de Integridade de Parâmetros**
- **type_safety**: Validação de tipos de parâmetros
- **bounds**: Verificação de ranges de hiperparâmetros
- **range_check**: Validação de valores válidos
- **real_arithmetic**: Verificação de aritmética real

### Exemplo de Saída de Verificação

```bash
INFO: 🔍 ✅ FORMAL VERIFICATION - CVC5
INFO:    📊 Contexto: DataPipeline → unknown
INFO:    🔧 Transformation: data_integrity_input_data
INFO:    ⚡ Status: SUCCESS (0.002s)
INFO:    📋 Constraints: 4/4 satisfied
INFO:    ✅ Satisfied: shape_preservation, type_safety, bounds, range_check
INFO:    💬 Z3: CVC5 verified all 4 constraints successfully
```

## 📊 Resultados e Logging

### Estrutura de Outputs

```bash
results/
├── solver_comparison/                    # Comparações multi-solver
├── datapipeline-*-verification-*.json   # Resultados verificação dados
├── labelpipeline-*-verification-*.json  # Resultados verificação labels  
├── z3-verification-*.json               # Resultados específicos Z3
└── *-complete-results-*.json            # Resultados completos experimentos

logs/
├── verification-*.json                  # Logs de verificação formal
└── experiments-*.log                    # Logs de execução de experimentos
```

### Métricas Coletadas

- **Performance**: Accuracy, F1-score, Precision, Recall
- **Verificação**: Número de constraints satisfeitos/violados, tempo de execução
- **Comparação**: Agreement entre solvers, diferenças de performance
- **Robustez**: Impacto das perturbações na precisão do modelo

## 🛠️ Comandos de Desenvolvimento

### Comandos de Qualidade
```bash
# Verificações completas de qualidade
make check                # Todas as verificações
make format              # Formatação automática (black + isort)
make lint                # Análise estática (pylint)
make test                # Testes unitários
```

### Comandos de Limpeza
```bash
make clean              # Remove cache Python
make clean-outputs      # Limpa logs/ e results/
make clean-venv         # Remove ambiente virtual
make clean-all          # Limpeza completa
```

### Comandos de Build e Deploy
```bash
make build              # Build do pacote
make publish            # Publica no TestPyPI
make ci                 # Pipeline completo CI/CD
```

## 🧪 Adicionando Novos Experimentos

### 1. Estrutura de um Experimento

Cada experimento segue o padrão estabelecido:

```python
# smart_inference_ai_fusion/experiments/meu_dataset/logistic_meu_dataset.py
from smart_inference_ai_fusion.core.experiment_runner import run_experiment_by_model
from smart_inference_ai_fusion.models.logistic_regression_model import LogisticRegressionModel
from smart_inference_ai_fusion.utils.types import SklearnDatasetName

def run():
    """Executa experimento LogisticRegression no dataset personalizado."""
    run_experiment_by_model(
        model_class=LogisticRegressionModel,
        dataset_name=SklearnDatasetName.MEU_DATASET,
        experiment_name="logistic_meu_dataset"
    )

if __name__ == "__main__":
    run()
```

### 2. Registrar no Registry

Adicione seu experimento no `experiment_registry.py`:

```python
# Definir experimentos do novo dataset
MEU_DATASET_EXPERIMENTS = {
    "LogisticRegressionModel": "smart_inference_ai_fusion.experiments.meu_dataset.logistic_meu_dataset",
    "DecisionTreeModel": "smart_inference_ai_fusion.experiments.meu_dataset.tree_meu_dataset", 
    "MLPModel": "smart_inference_ai_fusion.experiments.meu_dataset.mlp_meu_dataset",
}

# Mapear no registry principal
DATASET_EXPERIMENT_MAPPING = {
    # ... outros datasets
    "meu_dataset": MEU_DATASET_EXPERIMENTS,
}
```

### 3. Implementar Loader do Dataset

Se necessário, adicione função de carregamento em `sklearn_loader.py`:

```python
def _load_meu_dataset(self) -> MockBunch:
    """Carrega dataset personalizado."""
    # Implementar lógica de carregamento
    data, target = load_your_custom_data()
    
    return MockBunch(
        data=data,
        target=target,
        feature_names=[f"feature_{i}" for i in range(data.shape[1])],
        target_names=["class_0", "class_1"]  # ajustar conforme necessário
    )
```

### 4. Executar o Novo Dataset

```bash
# Testar com um solver
make run verify EXP=meu_dataset SOLVERS=z3

# Testar com ambos os solvers  
make run verify EXP=meu_dataset SOLVERS="z3,cvc5"

# Executar modelo específico
make run verify EXP=meu_dataset.logistic SOLVERS=z3
```
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

## 🚨 Troubleshooting

### Problemas Comuns

#### **1. Erro de Solver não Encontrado**
```bash
ERROR: Z3 solver not available
```
**Solução**: Instale as dependências SMT:
```bash
make install-dev  # Instala z3-solver e cvc5 automaticamente
```

#### **2. Timeout de Verificação**
```bash
WARNING: Verification timeout after 30.0s
```
**Solução**: Aumente o timeout:
```bash
VERIFICATION_TIMEOUT=60.0 make run verify EXP=wine SOLVERS=z3
```

#### **3. Erro de Registro de Experimento**
```bash
ERROR: Model 'LogisticRegressionModel' not found in registry
```
**Solução**: Verifique se o modelo está registrado em `experiment_registry.py`

#### **4. Problema de Memória em Datasets Grandes**
**Solução**: Use modelos otimizados para datasets grandes:
```python
# MLPModel já tem otimizações automáticas
# Para Adult dataset, usar max_iter=300 automaticamente aplicado
```

### Logs de Debug

Para debug detalhado:
```bash
# Ativar logging debug
LOG_LEVEL=DEBUG make run verify EXP=wine SOLVERS=z3

# Verificar logs específicos
tail -f logs/verification-*.json
tail -f logs/experiments-*.log
```

## 🤝 Contribuição

### Como Contribuir

1. **Fork** o repositório
2. **Clone** sua fork localmente
3. **Crie** uma branch para sua feature: `git checkout -b feature/nova-funcionalidade`
4. **Implemente** sua mudança seguindo os padrões do projeto
5. **Execute** os testes: `make check && make test`
6. **Commit** suas mudanças: `git commit -m "feat: adiciona nova funcionalidade"`
7. **Push** para sua branch: `git push origin feature/nova-funcionalidade`
8. **Abra** um Pull Request

### Padrões de Código

- **Formatação**: Black + isort (executar `make format`)
- **Linting**: Pylint com score > 8.0 (executar `make lint`)
- **Docstrings**: Estilo Google (verificar com `make style`)
- **Testes**: Pytest para novas funcionalidades
- **Commits**: Conventional Commits (`feat:`, `fix:`, `docs:`, etc.)

### Estrutura de Testes

```bash
tests/
├── unit/                    # Testes unitários
│   ├── test_models.py
│   ├── test_verification.py
│   └── test_datasets.py
├── integration/             # Testes de integração
│   ├── test_pipeline.py
│   └── test_multi_solver.py
└── fixtures/                # Dados de teste
    └── sample_data.py
```

## 📄 Licença

Este projeto está licenciado sob a **MIT License** - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🔗 Links Úteis

- **Documentação Z3**: [https://z3prover.github.io/api/html/index.html](https://z3prover.github.io/api/html/index.html)
- **Documentação CVC5**: [https://cvc5.github.io/docs/](https://cvc5.github.io/docs/)
- **Scikit-learn**: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
- **SMT-LIB**: [https://smtlib.cs.uiowa.edu/](https://smtlib.cs.uiowa.edu/)

## 📧 Contato

Para dúvidas, sugestões ou colaborações:

- **Autor**: Sergillam Barroso Oliveira
- **Email**: sgm.oliveira96@gmail.com
- **GitHub**: [@sergillam](https://github.com/sergillam)
- **LinkedIn**: [Sergillam](https://www.linkedin.com/in/sergillam-oliveira-0b6a9b8b/

---

**⭐ Se este projeto foi útil para você, considere dar uma estrela no repositório!**

---

*Desenvolvido como parte da pesquisa em Verificação Formal aplicada a Machine Learning - Universidade Federal do Amazonas - UFAM*
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
        logging.info("=== Executando todos os experimentos para <dataset> ===")
        run_<algoritmo>()  # Adicione mais funções se tiver outros experimentos
        logging.info("=== Experimentos concluídos ===")
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
