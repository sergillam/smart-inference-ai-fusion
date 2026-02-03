# Arquitetura do Sistema de Verificação no SIP

Este documento consolida os diagramas de arquitetura solicitados.

## Figura 1 — System Modules

```mermaid
graph TD
    A[CLI / Makefile] --> B[experiments/experiment_registry.py]
    B --> C[experiments/common.py<br/>run_standard_experiment]
    C --> D[datasets/<br/>factory + loaders]
    C --> E[models/<br/>wrappers sklearn]
    C --> F[inference/pipeline]
    F --> G[inference/engine(s)]
    G --> H[inference/transformations/*]
    F --> I[verification/core<br/>verification_manager]
    I --> J[verification/core/plugin_interface]
    J --> K[verification/solvers/z3_plugin]
    J --> L[verification/solvers/cvc5_plugin]
    I --> M[utils/solver_comparison.py]
    C --> N[utils/report.py<br/>report_data()]
    N --> O[(logs/)]
    N --> P[(results/)]
    subgraph Output
      O
      P
    end
    subgraph Verification
      I
      J
      K
      L
      M
    end
    subgraph Inference
      F
      G
      H
    end
    subgraph Data
      D
    end
    subgraph Models
      E
    end
```

## Figura 2 — Verification-augmented Experiment Flow

```mermaid
sequenceDiagram
    autonumber
    participant U as Usuário/Make
    participant R as ExperimentRegistry
    participant EXP as run_standard_experiment
    participant PIPE as InferencePipeline
    participant DATA as DatasetFactory
    participant VER as VerificationManager
    participant REG as VerifierRegistry
    participant Z3 as Z3 Plugin
    participant CVC5 as CVC5 Plugin
    participant REP as Report/Writer

    U->>R: solicitar execução (modelo,dataset,flags)
    R->>EXP: resolve config
    EXP->>DATA: load_data()
    DATA-->>EXP: X_train,X_test,y_train,y_test
    note over EXP: 1) Baseline<br/>2) Inference + Verificação
    EXP->>PIPE: construir pipeline
    PIPE->>PIPE: perturbar dados
    PIPE->>PIPE: perturbar rótulos
    PIPE->>PIPE: perturbar parâmetros
    alt Verificação habilitada
        PIPE->>VER: solicitar verificação
        VER->>REG: listar verificadores
        par Execução Multi-solver
            VER->>Z3: verify()
            VER->>CVC5: verify()
        end
        Z3-->>VER: raw result
        CVC5-->>VER: raw result
        VER-->>PIPE: veredictos
    end
    PIPE-->>EXP: dados & params finais
    EXP->>REP: salvar métricas + resultados
    REP-->>U: arquivos logs/ e results/
```

## Figura 3 — Multi-solver Verification Architecture

```mermaid
graph TD
    A[InferencePipeline] --> B[VerificationManager]
    B --> C[VerifierRegistry]
    C --> D[z3_plugin]
    C --> E[cvc5_plugin]
    subgraph Normalization & Comparison
      B --> F[result_normalizer]
      B --> G[solver_comparison]
    end
    D --> F
    E --> F
    F --> G
    G --> H[(logs/ + results/ unified)]
    B --> I[(circuit breaker / cache)]
    B --> J[(timeout handling)]
    style D fill:#f5f5f5,stroke:#666
    style E fill:#f5f5f5,stroke:#666
```

---

# Apêndice

## A. Pipeline Detail

```mermaid
flowchart LR
    subgraph Input
      A[X_train,X_test,y_train,y_test]
    end

    A --> B[DataNoiseConfig]
    B --> C[InferenceEngine (build pipeline)]
    C --> D[Transformações\nGaussian / Selective / Scale / Shuffle / Outliers / NaN / Remove / Swap / ClusterSwap / Drift]
    D --> E[X' (dados perturbados)]

    A --> F[LabelNoiseConfig]
    F --> G[LabelInferenceEngine]
    G --> H[Label Perturbações\nflip / confusion / partial / swap]
    H --> I[y' (labels perturbados)]

    A --> J[Param Config]
    J --> K[ParameterInferenceEngine]
    K --> L[Hyperparam Perturbações\ninteger / semantic / enum shift / scale]

    E --> M[Verificação pré/pós]
    H --> M
    L --> M
    M --> N[Veredictos]
    N --> O[Modelo + Métricas]
```

## B. Class Diagram Simplificado

```mermaid
classDiagram
    class Experiment {
        +run(X_train,X_test,y_train,y_test) dict
    }
    class BaseModel {
        +fit(X,y)
        +predict(X)
        +predict_proba(X) *
    }
    class InferencePipeline {
        +apply_data_inference()
        +apply_label_inference()
        +apply_param_inference()
    }
    class InferenceEngine {
        -pipeline : List[Transformation]
        +apply(X_train,X_test)
    }
    class VerificationManager {
        +verify(step, constraints, data)
    }
    class VerifierRegistry {
        +register(verifier)
        +get_enabled_solvers()
    }
    class FormalVerifier {
        <<interface>>
        +name
        +is_available()
        +verify(VerificationInput) VerificationResult
    }
    class Z3Plugin
    class CVC5Plugin
    class ReportUtil {
        +report_data(content,mode)
    }

    Experiment --> BaseModel
    Experiment --> InferencePipeline
    InferencePipeline --> InferenceEngine
    InferencePipeline --> VerificationManager
    VerificationManager --> VerifierRegistry
    VerifierRegistry --> FormalVerifier
    FormalVerifier <|-- Z3Plugin
    FormalVerifier <|-- CVC5Plugin
    Experiment --> ReportUtil
```

## C. Extension Points

```mermaid
graph TD
    A[Novos Datasets] --> B[datasets/factory]
    A2[Novos Model Wrappers] --> C[models/*_model.py]
    A3[Novas Transformações] --> D[inference/transformations]
    A4[Novos Solvers] --> E[verification/solvers/*]
    E --> F[VerifierRegistry]
    D --> G[InferenceEngine builder]
    C --> H[experiment_registry]
    B --> H
```

## D. Multi-Solver Sequence (Detalhado)

```mermaid
sequenceDiagram
    participant P as Pipeline Step
    participant M as VerificationManager
    participant VR as VerifierRegistry
    participant Z as Z3
    participant C as CVC5
    participant N as Normalizer
    participant CMP as Comparator

    P->>M: verify(constraints,input_data)
    M->>VR: get_active_solvers()
    VR-->>M: [Z3,CVC5]
    par Paralelo
      M->>Z: run()
      M->>C: run()
    end
    Z-->>M: raw_z3
    C-->>M: raw_cvc5
    M->>N: normalize ambos
    N-->>M: norm_z3,norm_cvc5
    M->>CMP: compare(norm_z3,norm_cvc5)
    CMP-->>M: parity metrics
    M-->>P: decisão consolidada
```

---
**Notas:**
- Todos os diagramas em Mermaid; podem ser renderizados em VS Code, GitHub ou MkDocs.
- Substitua legendas conforme estilo do artigo.
