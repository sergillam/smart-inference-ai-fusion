# Capítulo: Quantização para Modelos de Machine Learning

## 1. Introdução

A quantização é uma técnica fundamental na compressão e otimização de modelos de Machine Learning para implantação em dispositivos com recursos limitados. Este capítulo explora como a redução de precisão numérica afeta a robustez e eficiência de modelos de ML clássico, integrando esta técnica ao framework Smart Inference AI Fusion como módulo SIP-Q (Synthetic Inference Perturbations - Quantization).

### 1.1 Contexto de Pesquisa

Enquanto a maioria das pesquisas sobre quantização concentra-se em deep learning, o desempenho de algoritmos de ML clássico sob quantização permanece menos estudado. Este trabalho busca preencher essa lacuna investigando:

1. Como modelos clássicos (KNN, Decision Trees, MLP) degradam com precisão reduzida?
2. Qual é o ponto ótimo entre bits e acurácia para cada tipo de algoritmo?
3. Algoritmos não-supervisionados são mais ou menos sensíveis que supervisionados?
4. A quantização combinada com outras perturbações (SIP) é aditiva ou multiplicativa?

### 1.2 Motivação

**Implantação em Dispositivos Edge:** Dispositivos embarcados (microcontroladores, sensores IoT) suportam nativamente apenas `int8`, `int16` ou `float16`, não `float64`.

**Trade-off Acurácia vs Eficiência:** Reduzir a precisão dos dados diminui o uso de memória e o tempo de computação, mas aumenta o erro de predição.

**Pesquisa Aplicada:** Compreender esses trade-offs é essencial para engenheiros de ML que precisam implantar modelos em ambiente de recursos limitados.

---

## 2. Conceitos Fundamentais de Quantização

### 2.1 O que é Quantização?

Quantização é o processo de mapeamento de valores contínuos ou de alta precisão para um conjunto discreto de valores de baixa precisão. Formalmente:

$$Q(x) = \text{round}\left(\frac{x - \text{zero\_point}}{\text{scale}}\right)$$

Onde:
- $x$ é o valor original em precisão alta (ex: `float64`)
- $\text{zero\_point}$ é o offset que alinha o intervalo $[\text{min}, \text{max}]$ do dado com $[0, 2^b-1]$
- $\text{scale}$ é o fator de escala que normaliza o intervalo
- $b$ é a largura de bits alvo (8, 16 ou 32)

**Dequantização (Reconstrução):**

$$x_r = Q(x) \times \text{scale} + \text{zero\_point}$$

O valor reconstruído $x_r$ é uma aproximação de $x$, com erro definido como:

$$\varepsilon = |x - x_r|$$

### 2.2 Por Que Funciona?

A quantização explora dois princípios:

1. **Redundância de Precisão:** Muitos valores em precisão alta (`float64`) mapeiam para o mesmo inteiro de baixa precisão, sem impacto significativo no resultado final.

2. **Tolerância do Modelo:** Algoritmos de ML são, em geral, robustos a pequenas perturbações nos dados e parâmetros (até um certo limite).

**Exemplo Prático (Wine Dataset):**

```
Atributo: Alcohol (float64)
Função Original:     f(14.23, 1.71, 2.43, ...) → classe "A"
Quantizado (int8):   f(14.22, 1.72, 2.44, ...) → classe "A" ✓ Correto!
Erro introduzido:    ~0.01 por dimensão (imperceptível)
```

### 2.3 Tipos Numéricos em Quantização

O scikit-learn converte **todos os dados** para `float64` internamente, independente do tipo original:

| Tipo NumPy | Tamanho | Precisão | Papel |
|-----------|---------|----------|-------|
| `float64` | 8 bytes | 15-17 dígitos decimais | Baseline — precisão total |
| `float32` | 4 bytes | 6-9 dígitos decimais | Intermediário (não usado neste estudo) |
| `float16` | 2 bytes | 3-4 dígitos decimais | Meia precisão com alcance dinâmico |
| `int32` | 4 bytes | Inteiro exato (32 bits) | Quantização de baixa degradação (referência) |
| `int16` | 2 bytes | Inteiro exato (16 bits) | Quantização média — bom balanço |
| `int8` | 1 byte | Inteiro exato (8 bits) | Quantização agressiva — máxima compressão |

**Compressão Relativa (vs `float64`):**
- `float16` → 4× menor
- `int16` → 4× menor
- `int32` → 2× menor
- `int8` → 8× menor

### 2.4 Facto Fundamental: No scikit-learn, Tudo é float64

Antes de aprofundar a arquitetura de quantização, é essencial compreender um facto que condiciona toda a implementação:

> **O scikit-learn converte TODOS os dados numéricos para `float64` internamente.** Mesmo que o dataset original contenha inteiros (idade=25, quartos=3) ou floats de menor precisão, os métodos `fit()` e `predict()` operam sempre em `float64`.

Isto significa que **não existe distinção entre "dados originalmente inteiros" e "dados originalmente float"** no momento da quantização. O ponto de partida é **sempre `float64`** (64 bits, 8 bytes por valor).

```
Dados Originais (CSV/Dataset)        Após sklearn.load / pd.read_csv
┌──────────────────────────┐         ┌──────────────────────────┐
│ idade: 25    (int)       │         │ idade: 25.0   (float64)  │
│ salário: 3500 (int)      │ ──────► │ salário: 3500.0 (float64)│
│ altura: 1.75 (float)     │         │ altura: 1.75  (float64)  │
│ peso: 72.3   (float)     │         │ peso: 72.3    (float64)  │
│ classe: "A"  (str)       │         │ classe: 0     (int64)    │
└──────────────────────────┘         └──────────────────────────┘
                                      ↑
                                      Ponto de partida da quantização SIP-Q
                                      (SEMPRE float64 para features)
```

**Consequência prática:** Como o scikit-learn não aceita `int8` diretamente no `fit()`/`predict()`, o ciclo de quantização é sempre um **roundtrip**: `float64` → `intN` → `float64` (com erro). O modelo recebe dados `float64`, mas com os valores "degradados" — como se tivessem passado por um bottleneck de $b$ bits.

### 2.5 Quantização Por Grupo (Bit-Width Uniforme)

A quantização no SIP-Q é aplicada **por grupo**: num dado experimento, **todas as features recebem a mesma largura de bits**. Não se quantiza feature-a-feature com bits diferentes.

Cada feature tem os seus próprios parâmetros de calibração (`scale` e `zero_point`), calculados a partir do mínimo e máximo daquela coluna, mas o **tipo alvo é uniforme**.

**Exemplo concreto com Wine dataset (13 features) a 8-bit:**

| Feature | Range Original | scale | Quantizado |
|---------|---------------|-------|------------|
| Alcohol | [11.0, 14.8] | 0.0149 | int8 [0, 255] |
| Malic acid | [0.74, 5.80] | 0.0198 | int8 [0, 255] |
| Proline | [278, 1680] | 5.498 | int8 [0, 255] |

> **Nota:** O scale do Proline (5.498) é 370× maior que o do Alcohol (0.0149). Isto significa que a quantização int8 introduz 370× mais erro absoluto no Proline. Esta heterogeneidade de escala é intencional — mede a robustez real dos algoritmos.

---

## 3. Arquitetura de Quantização

### 3.1 Dois Contextos de Quantização

A quantização no SIP-Q opera sobre dois componentes distintos:

#### 3.1.1 Quantização de Dados (Features)

**O que é quantizado:** As features de entrada $X \in \mathbb{R}^{n \times p}$ (n amostras, p features).

**Como funciona:**

1. **Calibração (no conjunto de treino):**
   - Calcular $\min_i$ e $\max_i$ para cada feature $i$
   - Estimar $\text{scale}_i = \frac{\max_i - \min_i}{2^b - 1}$ (ex: 255 para 8 bits)
   - Estimar $\text{zero\_point}_i = \min_i$

2. **Transformação (em treino e teste com mesmos parâmetros):**
   - $X_{q[i,j]} = \text{clip}\left(\text{round}\left(\frac{X[i,j] - \text{zero\_point}_j}{\text{scale}_j}\right), 0, 2^b-1\right)$

3. **Dequantização:**
   - $X_r[i,j] = X_q[i,j] \times \text{scale}_j + \text{zero\_point}_j$

**Impacto:** O modelo recebe features com resolução reduzida, degradando a qualidade das distâncias e decisões.

#### 3.1.2 Quantização de Modelos (Parâmetros)

**O que é quantizado:** Pesos e parâmetros internos aprendidos pelo modelo.

| Algoritmo | Parâmetros Quantizados |
|-----------|------------------------|
| **KNN** | `_fit_X` (dados de treino armazenados) |
| **Decision Tree** | `tree_.threshold` (limiares de split) |
| **MLP** | `coefs_`, `intercepts_` (pesos e biases) |
| **MiniBatchKMeans** | `cluster_centers_` (centroides) |
| **GMM** | `means_`, `covariances_` (parâmetros gaussianos) |
| **Agglomerative** | Sem pesos (recalcula distâncias) |

**Processo:**

1. Treinar modelo normalmente em $X_{treino}$ com `float64`
2. Extrair parâmetros do modelo treinado
3. Aplicar quantização aos parâmetros
4. Dequantizar parâmetros e inserir novamente no modelo
5. Avaliar em $X_{teste}$

**Impacto:** O modelo "esquece" parte do conhecimento aprendido, simulando implantação em hardware de baixa precisão.

#### 3.1.3 Quantização Híbrida

**O que é:** Quantizar dados E parâmetros simultaneamente.

**Caso de uso real:** Um modelo implantado em dispositivo edge recebe inputs com `int8` e executa operações com pesos em `int8`.

**Hipótese:** A degradação combinada é multiplicativa, não apenas aditiva.

### 3.2 Fluxo Completo: Exemplo com Wine Dataset

```
┌────────────────────────────────────────────────────────────┐
│ ENTRADA: Wine dataset (13 features, 178 amostras)          │
│ Tipo: float64 | Memória: 178 × 13 × 8 bytes ≈ 18.5 KB    │
└────────────┬─────────────────────────────────────────────────┘
             │
             ├─────────────────────┬──────────────────────┐
             │                     │                      │
    ┌────────▼─────────┐  ┌────────▼────────┐  ┌─────────▼──────────┐
    │ BASELINE         │  │ Quantização int8 │  │ Quantização int16  │
    │ (float64)        │  │                  │  │                    │
    │ X_train: 13×8    │  │ scale/zp cal.   │  │ scale/zp cal.      │
    │ X_test: 52×8     │  │ X_q ∈ [0,255]   │  │ X_q ∈ [0,65535]    │
    │ model.fit()      │  │ X_r = reconstruído  │ X_r = reconstruído │
    │ acc_base: 98.1%  │  │ model.fit()     │  │ model.fit()        │
    └────────┬─────────┘  │ acc_q8: 96.2%   │  │ acc_q16: 97.8%     │
             │            └────────────────┘  └────────────────────┘
             │                    │                      │
             └────────────────────┴──────────────────────┘
                                  │
                      ┌───────────▼───────────┐
                      │ RESULTADO             │
                      │ Degradação int8: 1.9% │
                      │ Degradação int16: 0.3%│
                      └───────────────────────┘
```

---

## 4. Métodos de Quantização

Diferentes estratégias de quantização produzem diferentes níveis de erro. Este trabalho explora 4 métodos principais:

### 4.1 Quantização Uniforme (Uniform)

**Princípio:** Distribuir uniformemente os valores originais pelos níveis quantizados.

**Fórmula:**
$$\text{scale} = \frac{\max(X) - \min(X)}{2^b - 1}$$
$$\text{zero\_point} = \min(X)$$

**Vantagens:**
- Simplicidade: O(n)
- Determinístico: mesmo input sempre produz mesmo output
- Sem calibração complexa

**Desvantagens:**
- Outliers aumentam desnecessariamente o scale
- Distribuições não-uniformes podem ter resolução ruim em regiões densas

**Caso de Uso:** Dados normalizados ou com distribuição uniforme.

### 4.2 Quantização Min-Max (Min-Max)

**Princípio:** Quantização **assimétrica** que mapeia o intervalo real $[\min, \max]$ de cada feature individualmente para $[0, 2^b-1]$. O `zero_point` corresponde ao valor mínimo real da feature.

**Fórmula (por feature $j$):**
$$\text{scale}_j = \frac{\max(X_j) - \min(X_j)}{2^b - 1}$$
$$\text{zero\_point}_j = \min(X_j)$$

> **Diferença em relação à Uniforme:** A quantização uniforme (seção 4.1) calcula um **único** scale global para todo o array, assumindo distribuição simétrica em torno de zero. A Min-Max calcula **um par (scale, zero_point) por feature**, alinhando cada coluna individualmente ao intervalo inteiro $[0, 2^b-1]$. Isto evita desperdício de níveis quando as features têm ranges muito diferentes.

**Exemplo comparativo (Wine, Alcohol vs Proline):**

| | Uniform (scale global) | Min-Max (scale por feature) |
|---|---|---|
| Alcohol [11, 14.8] | Resolve bem (range pequeno) | scale = 0.015 |
| Proline [278, 1680] | Domina o scale global | scale = 5.498 |
| Resultado | Alcohol perde resolução | Cada feature otimizada |

**Vantagens:**
- Nenhum desperdício de níveis fora do range real de cada feature
- Calibração independente por coluna — robusto a diferenças de escala

**Desvantagens:**
- Outliers extremos numa feature comprimem os demais valores dessa coluna
- Necessita armazenar 2 parâmetros por feature (scale + zero_point)

**Caso de Uso:** Dados com ranges variados entre features (ex: Wine com 13 features em escalas muito diferentes).

### 4.3 Quantização K-Means (K-Means)

**Princípio:** Usar clustering k-means para encontrar centros ótimos dos níveis quantizados.

**Algoritmo:**
1. Executar k-means com $k = 2^b$ clusters
2. Os centros dos clusters tornam-se os níveis quantizados
3. Cada valor é mapeado para o cluster mais próximo

**Vantagens:**
- Erro muito baixo: distribui centroides baseado em densidade dos dados
- Ótimo para distribuições complexas/multimodais

**Desvantagens:**
- Complexidade: O(n × k × iterações) ≈ O(n × 256 × 20) = O(5000n)
- Convergência não-determinística
- k-means falha para $k > 256$ em datasets pequenos

**Caso de Uso:** Distribuições complexas; tempo de calibração não é crítico.

### 4.4 Quantização Percentile

**Princípio:** Usar percentis para encontrar limites de bins uniformes em escala quantilhada.

**Fórmula (exemplo com 8 bits = 256 níveis):**
$$\text{limites} = [p_0, p_{1/256}, p_{2/256}, \ldots, p_{255/256}]$$

**Vantagens:**
- Robustez: cada bin tem ~n/256 amostras
- Bom para outliers

**Desvantagens:**
- Complexidade: O(n log n)
- Bins não-uniformes no espaço original

**Caso de Uso:** Dados com distribuição desconhecida ou altamente não-uniforme.

### 4.5 Comparação de Métodos

| Método | Tipo | Complexidade | Erro Típico | Robustez a Outliers | Velocidade |
|--------|------|-------------|-----------|----------|-----------|
| Uniform | Simétrico | O(n) | Médio | Baixa | Muito rápida |
| Min-Max | Assimétrico | O(n) | Baixo | Baixa | Muito rápida |
| K-Means | Não-uniforme | O(n×k×i) | Muito Baixo | Alta | Lenta |
| Percentile | Não-uniforme | O(n log n) | Baixo | Alta | Rápida |

---

## 5. Quantização de Tipos: Integer vs Float16

### 5.1 Diferença Técnica Fundamental

Embora `int16` e `float16` ocupem ambos 2 bytes (4× compressão), seus comportamentos são fundamentalmente diferentes:

#### Quantização para `int16` (Inteiro)

**Processo sistemático:**
1. Calcular scale e zero_point baseado em min/max
2. Mapear valores continuamente distribuídos para 65.536 níveis discretos
3. Arredondar e clipar ao intervalo [0, 65535]
4. Guardar como inteiros de 16 bits

**Erro:** Uniforme dentro de cada bin

$$\varepsilon_{\text{uniforme}} \approx \frac{\text{scale}}{2}$$

**Vantagem:** Resolução previsível e controlada

```python
scale = (max_val - min_val) / 65535
X_q = np.clip(np.round((X - zero_point) / scale), 0, 65535).astype(np.uint16)
X_r = X_q.astype(np.float64) * scale + zero_point
```

#### Quantização para `float16` (IEEE 754 Half-Precision)

**Processo:** Cast direto — sem calibração

```python
X_q = X.astype(np.float16)
X_r = X_q.astype(np.float64)
```

**Características IEEE 754:**
- 1 bit de sinal
- 5 bits de expoente (range: $\approx 10^{-4.5}$ a $10^{4.5}$)
- 10 bits de mantissa (significância)
- **Resolução proporcional à magnitude:** erro relativo $\approx 2^{-10} \approx 0.1\%$

**Erro:** Proporcional à magnitude do valor

$$\varepsilon_{\text{relativo}} \approx 2^{-10} \times |x|$$

**Vantagem:** Melhor para valores de grande magnitude; maior alcance dinâmico

### 5.2 Quando Usar Cada Uma

```
┌────────────────────────────────────────────────────────┐
│            Atributo         │  int16    │  float16      │
├────────────────────────────────────────────────────────┤
│ Erro em pequenos valores   │ Grande % │ Pequena %     │
│ Erro em grandes valores    │ Pequena %│ Pequena %     │
│ Range dinâmico suportado   │ Fixo     │ Muito grande  │
│ Necessita calibração       │ Sim      │ Não           │
│ Determinístico             │ Sim      │ Sim           │
│ Hardware acelerado         │ Limitado │ GPUs/NPUs     │
└────────────────────────────────────────────────────────┘
```

**Recomendação:**
- **int16:** Dados normalizados em intervalo fixo (ex: pixels 0-255)
- **float16:** Dados com grande amplitude ou alcance dinâmico variável

---

## 6. Integração no Framework Smart Inference AI Fusion

### 6.1 Arquitetura Modular

O SIP-Q é implementado como módulo separado, mas integrado com SIP (perturbações) e SIP-V (verificação formal):

```
smart_inference_ai_fusion/
│
├── perturbation/        (SIP)   ← Perturbações sintéticas
├── verification/        (SIP-V) ← Verificação formal (Z3/CVC5)
└── quantization/        (SIP-Q) ← NOVO: Quantização
    ├── core/
    │   ├── config.py             # QuantizationConfig
    │   ├── types.py              # QuantMethod, BitWidth
    │   └── methods.py            # uniform, minmax, kmeans, percentile
    │
    ├── data/
    │   └── feature_quantizer.py  # float64 → int8/16/32/float16
    │
    ├── model/
    │   └── weight_quantizer.py   # Quantizar pesos de modelos
    │
    ├── hybrid/
    │   └── hybrid_quantizer.py   # Dados + Modelo simultaneamente
    │
    └── evaluation/
        ├── metrics.py            # Métricas supervisionadas/clustering
        └── benchmark.py          # Tempo, memória
```

### 6.2 Três Modos de Operação

Para cada experimento, escolher um modo:

| Modo | Quantizada | Impacto Medido | Questão Respondida |
|------|-----------|---|---|
| **data_only** | Apenas features $X$ | Degradação dos inputs | "Como reduzir bits de entrada?" |
| **model_only** | Apenas parâmetros | Degradação dos aprendizados | "Como comprimir modelos?" |
| **hybrid** | Ambos | Degradação combinada | "Como implantar em edge?" |

**Exemplo de uso:**

```python
from smart_inference_ai_fusion.quantization import QuantizationConfig, QuantizationExperiment

config = QuantizationConfig(
    data_bits=(8, 16, 32),
    model_bits=(8, 16, 32),
    method="uniform",
    enable_hybrid=True
)

experiment = QuantizationExperiment(config)
results = experiment.run(
    dataset="wine",
    algorithm="knn",
    modes=["data_only", "model_only", "hybrid"]
)

# results[mode][bits] → accuracy, error, time, memory
```

---

## 7. Questões de Pesquisa e Hipóteses

### 7.1 Questões de Pesquisa Formais

| ID | Questão |
|----|--------|
| **RQ1** | Qual é o impacto da quantização de dados (`uint8`/`uint16`/`uint32`) na acurácia de modelos supervisionados e na silhouette de modelos de clustering? |
| **RQ2** | Existem diferenças significativas de robustez à quantização entre algoritmos supervisionados e não-supervisionados? |
| **RQ3** | Qual é o bit-width mínimo que preserva a qualidade preditiva dentro de uma margem aceitável (≤ 5% de degradação)? |
| **RQ4** | O efeito combinado de perturbações SIP e quantização SIP-Q é aditivo ou existe interação entre os dois fatores? |

### 7.2 Hipóteses Estatísticas

| RQ | H₀ (Nula) | H₁ (Alternativa) | Teste Estatístico |
|----|-----------|-------------------|-------------------|
| **RQ1** | Não há diferença significativa entre a métrica baseline (`float64`) e após quantização a 16 bits | A quantização a 16 bits degrada significativamente a métrica | Wilcoxon signed-rank (pareado, 5 seeds) |
| **RQ2** | A degradação média dos algoritmos supervisionados é igual à dos não-supervisionados | Existe diferença significativa entre os dois grupos | Mann-Whitney U |
| **RQ3** | Todos os bit-widths (8, 16, 32) produzem degradação equivalente | Existe pelo menos um bit-width com degradação significativamente diferente | Friedman + post-hoc Nemenyi |
| **RQ4** | O efeito de SIP e SIP-Q combinados é a soma dos efeitos individuais (aditivo) | Existe interação entre SIP e SIP-Q (efeito não-aditivo) | ANOVA two-way |

> **Nota metodológica:** Testes não-paramétricos (Wilcoxon, Mann-Whitney, Friedman) são preferidos dado o número reduzido de repetições (5 seeds) e a ausência de garantia de normalidade. Correção de Holm-Bonferroni é aplicada nas comparações múltiplas (RQ3).

### 7.3 Critérios de Sucesso

#### Técnicos

| Critério | Alvo | Como Medir |
|----------|------|------------|
| Degradação em 16-bit | < 5% accuracy / < 0.05 silhouette | Média dos 5 seeds |
| Redução de memória em 8-bit | ≥ 75% (float64 → int8 = 8×) | `X.nbytes` antes/depois |
| Overhead da quantização | < 15% do tempo baseline | Mediana de 50 runs |
| Reprodutibilidade | 5 seeds | Seeds: 42, 123, 456, 789, 1024 |

#### De Investigação

| RQ | Validação |
|----|----------|
| **RQ1** | Ranking por degradação média + Wilcoxon pareado |
| **RQ2** | Mann-Whitney U entre grupos supervised/unsupervised |
| **RQ3** | Friedman + Nemenyi; ponto de inflexão (joelho) na curva |
| **RQ4** | ANOVA two-way em combinação com experimentos SIP |

---

## 8. Experimentos e Datasets

### 8.1 Datasets Selecionados

#### Supervisionados (Classificação)

| Dataset | Amostras | Features | Classes | Justificativa |
|---------|----------|----------|---------|---------------|
| **Wine** | 178 | 13 | 3 | Features contínuas com ranges muito variados (Alcohol ∈ [11,15] vs. Proline ∈ [278,1680]). Teste ideal de sensibilidade da quantização a escalas diferentes. |
| **Digits** | 1797 | 64 | 10 | Features de imagem (pixels 0-16), alta dimensionalidade. Domínio mais relevante para quantização na prática — imagens são naturalmente quantizadas. |

#### Não-Supervisionados (Clustering)

| Dataset | Amostras | Features | Clusters | Justificativa |
|---------|----------|----------|----------|---------------|
| **Make Blobs** | 500 | 2 | 3 | Clusters isotrópicos (esféricos). Baseline controlado — se a quantização falha aqui, há um problema fundamental. |
| **Make Moons** | 500 | 2 | 2 | Clusters não-lineares (forma de lua). Testa se a quantização destrói a estrutura não-linear dos dados. |

**Cobertura do design:**

```
                    Dimensionalidade
                    Baixa (2D)       Média (13D)     Alta (64D)
                +----------------+---------------+---------------+
  Supervisado   |                |   Wine [x]    |   Digits [x]  |
                +----------------+---------------+---------------+
  Nao-Superv.   | Make Blobs [x] |               |               |
                | Make Moons [x] |               |               |
                +----------------+---------------+---------------+
```

### 8.2 Algoritmos Testados e Sensibilidade à Quantização

#### Supervisionados (3)

| Algoritmo | Sensibilidade | Parâmetros Quantizados | Justificativa |
|-----------|--------------|------------------------|---------------|
| **K-Nearest Neighbors (KNN)** | **Alta** | `_fit_X` (dados de treino armazenados) | Baseia-se em distâncias euclidianas — pequenas alterações mudam quais são os vizinhos mais próximos. Compressão máxima de memória. |
| **Decision Tree (DT)** | **Baixa** | `tree_.threshold` (limiares de split) | Usa limiares discretos. A quantização só afeta se o arredondamento mover um valor para o lado errado do limiar. Funciona como "controlo negativo". |
| **MLP** | **Alta** | `coefs_`, `intercepts_` (pesos e biases) | Maior número de parâmetros quantizáveis. Serve de ponte com a literatura de deep learning quantization. |

#### Não-Supervisionados (3)

| Algoritmo | Sensibilidade | Parâmetros Quantizados | Justificativa |
|-----------|--------------|------------------------|---------------|
| **MiniBatchKMeans** | **Média** | `cluster_centers_` (centróides) | Centróides são médias dos pontos — alguma tolerância ao ruído. Mas atribuição por distância é sensível. |
| **GMM** | **Alta** | `means_`, `covariances_` | Opera com médias E covariâncias. Covariâncias envolvem valores muito pequenos e diferenças sutis — muito sensível. |
| **Agglomerative Clustering** | **Média** | Sem pesos internos (recalcula distâncias) | Apenas quantização dos dados de entrada tem efeito (`model_only` equivale ao baseline). Abordagem fundamentalmente diferente. |

**Perfil de sensibilidade esperado:**

```
Alta      │  MLP  ====================  (muitos pesos)
          │  GMM  ==================    (covariâncias sensíveis)
          │  KNN  ================      (distâncias afetadas)
Média     │  MBK  ==========           (centróides tolerantes)
          │  AC   ========             (recalcula distâncias)
Baixa     │  DT   ====                 (limiares discretos)
          └────────────────────────────
```

### 8.3 Matriz Datasets × Algoritmos

| | Wine (13D, 3c) | Digits (64D, 10c) | Make Blobs (2D, 3c) | Make Moons (2D, 2c) |
|---|---|---|---|---|
| **KNN** | ✓ | ✓ | — | — |
| **DT** | ✓ | ✓ | — | — |
| **MLP** | ✓ | ✓ | — | — |
| **MiniBatchKMeans** | — | — | ✓ | ✓ |
| **GMM** | — | — | ✓ | ✓ |
| **Agglomerative** | — | — | ✓ | ✓ |

**Total de combinações únicas: 12**

### 8.4 Design Experimental e Volume de Dados

Estrutura de execução para cada combinação (dataset, algoritmo):

```
Para cada (dataset, algoritmo, bit_width, seed):
    ├─ BASELINE (float64)
    │   X_train (float64) → model.fit() → model.predict(X_test) → acc_base
    │
    ├─ data_only (int8/16/32)
    │   X_train → quantize → dequantize → model.fit() → acc_data
    │
    ├─ model_only (int8/16/32)
    │   model.fit(X_train) → quantize pesos → model'.predict() → acc_model
    │
    └─ hybrid (int8/16/32)
        X → quantize → model.fit() → quantize pesos → model'.predict() → acc_hybrid
```

**Parâmetros do protocolo:**

| Parâmetro | Valores | Total |
|-----------|---------|-------|
| Combinações dataset×algoritmo | 12 | — |
| Bit-widths | 8, 16, 32 | 3 |
| Modos | data_only, model_only, hybrid | 3 |
| Seeds | 42, 123, 456, 789, 1024 | 5 |
| Split treino/teste | 70/30 | — |

**Volume total:**
- **Execuções quantizadas:** 12 × 3 × 3 × 5 = **540 execuções**
- **Baselines:** 12 × 5 = **60 execuções**
- **Total geral:** **600 medições**
- **Extensão float16 opcional:** +60 a 180 medições adicionais

---

## 9. Resultados Esperados e Análise

### 9.1 Métricas Supervisionadas

Para classificação:
- **Acurácia:** $\frac{\text{corretos}}{\text{total}}$
- **Precisão e Recall:** por classe
- **F1-Score:** balanço P/R

### 9.2 Métricas de Clustering

Para clustering:
- **Silhueta:** coesão dentro/separação entre clusters
- **Davies-Bouldin:** distância intra/inter-cluster
- **Índice de Calinski-Harabasz:** razão variância entre/dentro

### 9.3 Benchmarks de Eficiência

- **Tempo de execução:** fit/predict com/sem quantização
- **Uso de memória:** tamanho dos modelos antes/depois
- **Trade-off:** curvas Pareto acurácia vs compressão

### 9.4 Visualizações Esperadas

```
1. Heatmap: Acurácia vs (algoritmo × bit-width) — Wine dataset
   ┌──────────────────────────────────────────────────────┐
   │ Algoritmo │ 8-bit │ 16-bit │ 32-bit │ Sensibilidade │
   ├───────────┼───────┼────────┼────────┼───────────────┤
   │ DT        │ 97%   │ 98%    │ 98%    │ Baixa         │
   │ KNN       │ 92%   │ 97%    │ 98%    │ Alta          │
   │ MLP       │ 85%   │ 94%    │ 97%    │ Alta          │
   └───────────┴───────┴────────┴────────┴───────────────┘
   Nota: DT degrada pouco mesmo em 8-bit (limiares discretos).
   MLP degrada mais (muitos pesos sensíveis).

2. Curvas: Acurácia vs Bits (por algoritmo)
   Acurácia │
        100 │  DT──o────o────o (baseline ≈ quant)
         95 │     KNN──o────o
         90 │        ╱
         85 │  MLP──o          ← "joelho" em 16-bit
         80 └──────────────────
            8       16       32  (bits)

3. Fronteira Pareto: Acurácia vs Compressão
   Acurácia │
        100 │ ✓✓✓ (int32)
         95 │  ✓✓ (int16) ← região viável
         90 │   ✓ (int8)
            └─────────────────
              2×   4×   8×  (compressão)
```

---

## 10. Decisões Metodológicas

### 10.1 Normalização vs. Quantização Direta

Quando as features têm escalas muito diferentes (exemplo — Wine: Alcohol ∈ [11,15] vs. Proline ∈ [278,1680]), a quantização uniforme distribui os níveis de forma desigual entre features.

**Decisão adotada:** Quantizar os dados **sem normalização prévia**, para medir o impacto real da quantização nos dados tal como existem.

**Justificativa:** A normalização (StandardScaler / MinMaxScaler) é uma transformação independente. Combiná-la com a quantização confundiria dois efeitos distintos. O objetivo do SIP-Q é medir a robustez dos algoritmos ao erro de quantização, não otimizar a acurácia.

> **Exceção:** O método `minmax_quantize` inclui normalização implícita (mapeia para [0,1] antes de quantizar). Isto é explicitamente reportado nos resultados como variante metodológica.

> **Nota para a dissertação:** Esta decisão deve ser justificada na secção de Metodologia, referenciando a prática padrão em benchmarks de quantização (ex: TensorFlow Lite quantiza pesos e ativações no espaço original, não normalizado).

### 10.2 O Roundtrip float64 → int → float64

Um aspeto fundamental da implementação é que o scikit-learn não aceita `int8` diretamente nos métodos `fit()`/`predict()`. O ciclo completo é sempre:

$$\text{float64} \xrightarrow{\text{quantizar}} \text{int}N \xrightarrow{\text{dequantizar}} \text{float64 (com erro)}$$

Isto é uma **simulação** de erro de quantização, não uma implementação de inferência em hardware. O objetivo é medir a robustez algorítmica — como os algoritmos se comportam quando os dados têm resolução limitada — e não medir speedup de hardware.

### 10.3 Dados Originalmente Inteiros

Alguns datasets têm features que já são essencialmente inteiras em `float64` (ex: Digits, com pixels 0-16). Para esses dados:

```
Digits — Pixel com valor 12 (armazenado como float64):

float64:  12.000000000000000   (8 bytes, 64 bits de precisão)
     │
     ├──► int8:   12  (valor exato — 256 níveis capturam 17 valores distintos)
     ├──► int16:  12  (valor exato)
     └──► int32:  12  (valor exato)

A quantização NÃO introduz erro nesses casos!
```

**Consequência:** Espera-se que Digits apresente **menos degradação** que Wine, porque 256 níveis (int8) capturam perfeitamente os 17 valores distintos dos pixels (0-16). Esta diferença entre datasets é em si um resultado de pesquisa interessante.

---

## 11. Ameaças à Validade

### 11.1 Validade Interna

| Ameaça | Mitigação |
|--------|----------|
| **Vazamento de dados na calibração** | `FeatureQuantizer.fit()` calibrado apenas no treino; `transform()` aplicado separadamente ao teste com os mesmos parâmetros. |
| **Parâmetros de quantização dependentes do seed** | O seed controla apenas o split treino/teste e inicialização de modelos estocásticos. Os parâmetros de quantização derivam deterministicamente dos dados de treino. |
| **Overfitting do método ao dataset** | Quatro métodos independentes (uniform, min-max, k-means, percentile) reduzem risco de conclusão espúria. |

### 11.2 Validade Externa

| Ameaça | Mitigação |
|--------|----------|
| **Poucos datasets (4)** | Representam 4 cenários distintos: alta dimensionalidade (Digits, 64D), média com escala variada (Wine, 13D), sintéticos bem-separados (Blobs) e fronteira não-linear (Moons). |
| **Apenas algoritmos clássicos** | Escopo deliberado: o framework SIP foca em ML clássico. Deep learning está fora do escopo. |
| **Ausência de dados tabulares reais de grande escala** | Limitação reconhecida. Trabalho futuro pode incluir datasets UCI maiores. |

### 11.3 Validade de Construto

| Ameaça | Mitigação |
|--------|----------|
| **Roundtrip não simula hardware real** | Explicitamente documentado como simulação de erro de quantização. O objetivo é medir robustez algorítmica, não performance de hardware. |
| **sklearn converte tudo para float64** | Medição de overhead inclui ciclo completo (quantize + dequantize + inferência), reportado como `overhead_pct`, não como speedup absoluto. |

### 11.4 Validade de Conclusão

| Ameaça | Mitigação |
|--------|----------|
| **Múltiplas comparações** | Correção de Holm-Bonferroni aplicada quando se comparam mais de 2 condições (RQ3). |
| **Tamanho amostral reduzido (5 seeds)** | Testes não-paramétricos usados. Effect size (Cohen's d) reportado em todas as comparações. |

---

## 12. Conclusões Parciais

### 12.1 Achados Esperados

1. **Robustez decrescente com bits:** Todos os algoritmos mostram degradação previsível, mas em graus diferentes
2. **Decision Tree como controlo negativo:** DT deve apresentar a menor degradação, pois seus limiares discretos são robustos a pequenas alterações
3. **GMM e MLP como casos mais sensíveis:** Alta densidade paramétrica e covariâncias sensíveis devem apresentar maior degradação
4. **Ponto ótimo em 16-bit:** Para a maioria dos algoritmos, int16 oferece o equilíbrio ideal entre compressão e qualidade
5. **Digits mais robusto que Wine em 8-bit:** Features pseudo-inteiras degradam menos que features verdadeiramente contínuas
6. **Híbrida não é 2×:** A perda combinada é sub-aditiva — os modelos absorvem parte do erro

### 12.2 Implicações Práticas

- **Deployment em IoT:** int16 (4× compressão) é viável para >95% dos casos com degradação < 5%
- **Compressão máxima:** int8 (8× compressão) aceitável para DT e MiniBatchKMeans
- **Trade-off recomendado:** 4× compressão (int16) por ~1-2% de acurácia
- **float16 vs int16:** Comparação depende do perfil dos dados — int16 é mais previsível, float16 é melhor para grandes magnitudes

---

## 13. Referências e Trabalhos Relacionados

### 13.1 Quantização em Deep Learning

- **Jacob et al. (2018):** "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" — *IEEE CVPR 2018*. Formaliza a quantização assimétrica com scale e zero_point; base teórica da seção 2.1.
- **Gong et al. (2014):** "Compressing Deep Convolutional Networks using Vector Quantization" — *arXiv 1412.6115*. Introduz k-means para quantização de pesos; precursor do método k-means descrito na seção 4.3.
- **Han et al. (2016):** "Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding" — *ICLR 2016*. Mostra que MLP tolera int8 com degradação controlada; relevante para a hipótese de sensibilidade da seção 8.2.

### 13.2 Quantização em ML Clássico (Lacuna)

Em contraste com deep learning, a literatura de quantização para modelos de ML clássico é escassa. Trabalhos como **Gupta et al. (2015)** e **Courbariaux et al. (2015)** focam exclusivamente em redes neurais. O presente trabalho posiciona-se nesta lacuna, investigando KNN, DT, MLP, KMeans, GMM e Agglomerative Clustering — algoritmos com estruturas radicalmente diferentes que reagem de formas distintas à redução de precisão.

### 13.3 Robustez de Modelos

- **Goodfellow et al. (2014):** "Explaining and Harnessing Adversarial Examples" — Establece o conceito de perturbação controlada para avaliar robustez; precursor conceitual do SIP.
- **Szegedy et al. (2013):** "Intriguing Properties of Neural Networks" — Primeira demonstração de que redes neurais são sensíveis a perturbações imperceptíveis; motivação para estudar robustez em modelos clássicos.

### 13.4 Verificação Formal de ML

- **Katz et al. (2017):** "Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks" — Base do módulo SIP-V com Z3/CVC5 que se integra com SIP-Q.
- **Ehlers (2014):** "Formal Verification of Piece-Wise Linear Feed-Forward Neural Networks" — Verificação de propriedades de segurança em redes neurais.

### 13.5 Padrões e Frameworks de Quantização

- **TensorFlow Lite (Google, 2019):** Post-training quantization para deployment em edge — quantiza pesos no espaço original (sem normalização prévia), alinhado com a decisão metodológica da seção 10.1.
- **PyTorch Quantization (Facebook, 2020):** Suporte nativo a `int8` por meio de `torch.quantization` — referência para as fórmulas da seção 2.1.

---

## Apêndice: Código Representativo

### A.1 Quantização Uniforme (Pseudocódigo)

```python
class FeatureQuantizer:
    def __init__(self, bits=8, method="uniform"):
        self.bits = bits
        self.method = method
        self.scale_ = None
        self.zero_point_ = None

    def fit(self, X):
        # Calibração apenas no treino
        min_val = X.min(axis=0)
        max_val = X.max(axis=0)

        self.scale_ = (max_val - min_val) / (2**self.bits - 1)
        self.zero_point_ = min_val
        return self

    def transform(self, X):
        # Aplicar transformação
        X_norm = (X - self.zero_point_) / self.scale_
        X_q = np.clip(np.round(X_norm), 0, 2**self.bits - 1)
        return X_q.astype(np.uint8) if self.bits == 8 else X_q

    def inverse_transform(self, X_q):
        # Reconstruir
        return X_q * self.scale_ + self.zero_point_
```

### A.2 Experimento Completo

```python
quantizer = FeatureQuantizer(bits=16, method="uniform")

# Treino
quantizer.fit(X_train)
X_train_q = quantizer.transform(X_train)
X_train_r = quantizer.inverse_transform(X_train_q)

# Teste
X_test_q = quantizer.transform(X_test)
X_test_r = quantizer.inverse_transform(X_test_q)

# Avaliação
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_r, y_train)

acc_quantized = model.score(X_test_r, y_test)
print(f"Acurácia (int16): {acc_quantized:.2%}")
print(f"Degradação: {(acc_baseline - acc_quantized):.2%}")
```

### A.3 Quantização de Parâmetros do Modelo

```python
import copy
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def quantize_model_weights(model, bits=8):
    """Quantiza parâmetros internos de um modelo treinado."""
    model_q = copy.deepcopy(model)
    quantizer = FeatureQuantizer(bits=bits)

    if isinstance(model_q, KNeighborsClassifier):
        # KNN armazena dados de treino em _fit_X
        X_fit = model_q._fit_X
        quantizer.fit(X_fit)
        X_q = quantizer.transform(X_fit)
        model_q._fit_X = quantizer.inverse_transform(X_q)

    return model_q

# Uso
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

model_q = quantize_model_weights(model, bits=16)
acc_model_quant = model_q.score(X_test, y_test)
print(f"Degradação model_only: {(acc_baseline - acc_model_quant):.2%}")
```

---

**Documento compilado para uso como capítulo de TCC**

*Nota: Este documento sintetiza a técnica de quantização, sua aplicação teórica e prática no framework SIP-Q, adequado para contextualizar a pesquisa em um trabalho de conclusão de curso de Engenharia, Ciência da Computação ou Inteligência Artificial.*
