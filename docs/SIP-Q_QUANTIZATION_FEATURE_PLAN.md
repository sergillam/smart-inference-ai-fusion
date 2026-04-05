# SIP-Q: Extensão de Quantização para Smart Inference AI Fusion

**Documento de Planejamento de Feature**
**Versão:** 4.1 (Versão Final Revista)
**Data:** 4 de Março de 2026
**Status:** ✅ Aprovado para Implementação

---

## 📋 Índice

1. [Resumo Executivo](#resumo-executivo)
2. [Nomenclatura](#nomenclatura)
3. [Motivação](#motivação)
4. [Arquitetura e Sistema de Tipos](#arquitetura-e-sistema-de-tipos)
5. [Fases de Implementação](#fases-de-implementação)
6. [Seleção de Datasets](#seleção-de-datasets)
7. [Seleção de Algoritmos](#seleção-de-algoritmos)
8. [Questões de Pesquisa e Hipóteses](#questões-de-pesquisa-e-hipóteses)
9. [Critérios de Sucesso](#critérios-de-sucesso)
10. [Ameaças à Validade](#ameaças-à-validade)
11. [Decisões Metodológicas](#decisões-metodológicas)
12. [Próximos Passos](#próximos-passos)
13. [Referências](#referências)

---

## 🎯 Resumo Executivo

**SIP-Q (Synthetic Inference Perturbations - Quantization)** é uma extensão do framework
Smart Inference AI Fusion que introduz quantização controlada como técnica de perturbação
para avaliar robustez e eficiência de modelos supervisionados e não-supervisionados.

### Objetivos

- Avaliar como a redução de precisão numérica afeta modelos de ML clássico
- Medir o trade-off entre acurácia e eficiência computacional
- Comparar robustez à quantização entre 6 algoritmos (3 supervisionados + 3 não-supervisionados)
- Testar em 4 datasets representativos (2 supervisionados + 2 não-supervisionados)
- Integrar com os frameworks SIP (perturbações) e SIP-V (verificação formal) existentes

### Resultados Esperados

| Entregável | Descrição |
|------------|-----------|
| **4 métodos** | Quantização uniforme, min-max, k-means e percentile |
| **4 datasets** | Wine, Digits, Make Blobs, Make Moons |
| **6 algoritmos** | KNN, DT, MLP + MiniBatchKMeans, GMM, Agglomerative |
| **Baseline obrigatório** | Execução normal (`float64`) para cada par dataset×algoritmo×seed |
| **Trilha mista opcional** | `float16` para comparar 16-bit inteiro vs 16-bit ponto flutuante |
| **1 case study** | `case4.py` — pipeline automatizado de quantização |
| **Análise** | Curvas acurácia vs. bits, fronteiras de Pareto |

---

## 🏷️ Nomenclatura

### SIP-Q — Synthetic Inference Perturbations - Quantization

```
Smart Inference AI Fusion
│
├─ SIP (Perturbações de Inferência Sintética)
│  ├─ Perturbações de Dados (19 tipos)
│  ├─ Perturbações de Parâmetros (10 tipos)
│  └─ Perturbações de Rótulos (5 tipos)
│
├─ SIP-V (Verificação Formal)
│  ├─ Z3 Solver
│  ├─ CVC5 Solver
│  └─ Comparação de Solvers
│
└─ SIP-Q (Quantização) ← NOVO
   ├─ Quantização de Dados
   ├─ Quantização de Modelos
   └─ Quantização Híbrida (Dados + Modelo)
```

---

## 💡 Motivação

### Por que Quantização?

**Implantação real:** Dispositivos edge (int8/int16/float16) não suportam float64.
**Lacuna na literatura:** A maior parte da pesquisa foca em deep learning; ML clássico é pouco estudado.
**Questões de pesquisa abertas:**

1. Como modelos clássicos (KNN, DT, GMM) degradam com precisão reduzida?
2. Qual é o ponto ótimo entre bits e acurácia para cada tipo de algoritmo?
3. Algoritmos não-supervisionados são mais ou menos sensíveis que supervisionados?
4. A quantização combinada com perturbações SIP é aditiva ou multiplicativa?

### Relação com SIP e SIP-V

```
SIP   → Testa robustez via perturbações sintéticas
SIP-V → Verifica correção via métodos formais (Z3/CVC5)
SIP-Q → Avalia degradação de precisão via quantização

COMBINADO → SIP + SIP-V + SIP-Q = Análise completa de robustez, correção e eficiência
```

---

## 🏗️ Arquitetura e Sistema de Tipos

### Sistema de Tipos para Quantização

A quantização opera sobre a transformação de tipos numéricos. É fundamental entender
quais tipos de dados existem no pipeline e como cada um é afetado:

#### Tipos de Dados Envolvidos

| Tipo NumPy | Tamanho | Papel no SIP-Q | Onde Aparece |
|------------|---------|-----------------|--------------|
| `float64` | 8 bytes | **Tipo original** — baseline do scikit-learn | Features (X), pesos de modelo, centroides |
| `float32` | 4 bytes | Tipo intermédio do NumPy (fora do escopo SIP-Q) | Não é alvo de quantização neste estudo |
| `float16` | 2 bytes | **Half-precision** — trilha de extensão | Dados e pesos quantizados (cast direto) |
| `int32` | 4 bytes | **Inteiro largo** — referência quantizada (2×) | Features e pesos quantizados |
| `int16` | 2 bytes | **Quantização média** — boa relação custo/benefício (4×) | Features e pesos quantizados |
| `int8` | 1 byte | **Quantização agressiva** — máxima compressão (8×) | Features e pesos quantizados |

> **Nota:** Tipos `str` e `object` não participam da quantização. Features categóricas
> (strings) devem ser previamente codificadas (one-hot, label encoding) antes de quantizar.
> A quantização opera exclusivamente sobre dados numéricos contínuos.

#### Fluxo de Conversão de Tipos

```
                    ENTRADA                      QUANTIZAÇÃO                    SAÍDA
             ┌──────────────────┐         ┌─────────────────────┐      ┌──────────────────┐
             │                  │         │   uniform_quantize  │      │                  │
             │   X: float64    ─┼────────►│   1. Calcular scale ├─────►│   X_q: int8      │
             │   (8 bytes/val)  │         │   2. Calcular zero  │      │   (1 byte/val)   │
             │                  │         │   3. clip + round   │      │   Compressão: 8× │
             └──────────────────┘         └─────────────────────┘      └──────────────────┘
                                                                               │
                                          ┌─────────────────────┐              │
                                          │    dequantize       │              │
             ┌──────────────────┐         │   1. Subtrair zero  │◄─────────────┘
             │                  │◄────────┤   2. Multiplicar    │
             │   X_r: float64  │ (c/ erro)│      por scale     │
             │   (reconstruído) │         └─────────────────────┘
             └──────────────────┘

             Erro de quantização = ||X - X_r||
```

#### Tabela de Conversão por Largura de Bits

```
float64 (64 bits) ──┬── quantizar para int32 (32 bits)  → 2× compressão
                    ├── quantizar para int16 (16 bits)  → 4× compressão, 65536 níveis
                    ├── quantizar para float16 (16 bits)→ 4× compressão, maior alcance dinâmico
                    └── quantizar para int8  (8 bits)   → 8× compressão, 256 níveis
```

#### O que é Quantizado em Cada Contexto

**1. Quantização de Dados (features X):**

| Componente | Tipo Original | Tipo Quantizado | Impacto |
|------------|---------------|-----------------|---------|
| Features contínuas | `float64` | `int8`/`int16`/`float16`/`int32` | Perda de resolução nas distâncias |
| Labels (classificação) | `int64` | Não quantizado | São discretos, não faz sentido |
| Labels (clustering) | `int64` | Não quantizado | Atribuições de cluster |

**2. Quantização de Modelos (pesos internos):**

| Modelo | Atributo Quantizado | Tipo Original | Tipo Quantizado |
|--------|---------------------|---------------|-----------------|
| **KNN** | `_fit_X` (dados de treino armazenados) | `float64` | `int8`/`int16` |
| **Decision Tree** | `tree_.threshold` (limiares de split) | `float64` | `int16`/`float16` |
| **MLP** | `coefs_`, `intercepts_` (pesos e biases) | `float64` | `int8`/`float16` |
| **MiniBatchKMeans** | `cluster_centers_` (centroides) | `float64` | `int16`/`float16` |
| **GMM** | `means_`, `covariances_` | `float64` | `int16`/`int32` |
| **Agglomerative** | Sem pesos internos (recalcula distâncias) | — | Quantizar X |

> **Nota sobre Agglomerative Clustering em `model_only`:** Como o Agglomerative não armazena
> pesos internos (recalcula a árvore de distâncias a cada chamada), o modo `model_only`
> produzirá resultado idêntico ao baseline. Isto é esperado e documentado — o modo `data_only`
> e `hybrid` são equivalentes para este algoritmo.

### Estratégia de Quantização: O Que Quantizamos e Como

#### Resposta Direta: Sim, Quantizamos AMBOS — Dados e Parâmetros

O SIP-Q opera em **3 modos independentes** para cada experimento:

| Modo | O que é quantizado | Objetivo |
|------|-------------------|----------|
| **`data_only`** | Features do dataset (X) | Medir impacto da redução de precisão nos dados de entrada |
| **`model_only`** | Pesos internos do modelo treinado | Medir impacto da redução de precisão nos parâmetros aprendidos |
| **`hybrid`** | Dados + Modelo simultaneamente | Medir se a degradação combinada é aditiva ou multiplicativa |

#### Escopo de Precisão Numérica

- **Core obrigatório do estudo:** `int8`, `int16`, `int32`.
- **Trilha opcional recomendada:** `float16` (comparação 16-bit inteiro vs 16-bit float).

Justificativa para incluir `float16`:

1. Mesmo custo de memória de 16-bit (4x compressão vs `float64`), mas comportamento numérico diferente.
2. Representa melhor hardware moderno com suporte nativo a half-precision.
3. Permite responder uma pergunta prática: **em 16 bits, inteiro ou ponto flutuante é mais robusto por algoritmo?**

#### Diferença Técnica: Quantização Inteira vs. float16

A quantização para tipos inteiros e para `float16` são operações **fundamentalmente diferentes**:

| Aspecto | Quantização Inteira (int8/16/32) | Quantização float16 |
|---------|----------------------------------|----------------------|
| **Operação** | `scale + zero_point + clip + round` | Cast direto: `data.astype(np.float16)` |
| **Parâmetros** | `scale`, `zero_point` por feature | Nenhum — conversão nativa do IEEE 754 |
| **Dequantização** | `quantized * scale + zero_point` | Cast inverso: `data.astype(np.float64)` |
| **Erro** | Uniforme dentro de cada bin | Proporcional à magnitude (mantissa de 10 bits) |
| **Overflow** | Clipado ao range [0, 2^N-1] | Saturação ±65504 (max float16) |
| **Valores sub-normais** | Não se aplica | Perda de precisão perto de zero |

```python
# Quantização inteira (int16): requer calibração
scale = (max_val - min_val) / 65535
zero_point = min_val
X_q = np.clip(np.round((X - zero_point) / scale), 0, 65535).astype(np.uint16)
X_r = X_q.astype(np.float64) * scale + zero_point   # reconstrução

# Quantização float16: cast direto (sem calibração)
X_q = X.astype(np.float16)
X_r = X_q.astype(np.float64)   # reconstrução — mais simples
```

**Consequência prática:** `float16` preserva melhor valores de grande magnitude mas
perde precisão em valores muito pequenos. `int16` distribui o erro uniformemente
pelo range, independente da magnitude.

#### Facto Fundamental: No scikit-learn, TUDO é float64

Antes de explicar como a quantização funciona, é essencial entender um facto:

> **O scikit-learn converte TODOS os dados numéricos para `float64` internamente.**
> Mesmo que o CSV original tenha inteiros (idade=25, quartos=3) ou floats menores,
> o `fit()` e `predict()` do scikit-learn operam sempre em `float64`.

Isto significa que **não existe distinção entre "dados int" e "dados float"** no momento
da quantização. O ponto de partida é **sempre `float64`** (64 bits, 8 bytes por valor).

```
Dados Originais (CSV/Dataset)        Após sklearn.load / pd.read_csv
┌──────────────────────────┐         ┌──────────────────────────┐
│ idade: 25    (int)       │         │ idade: 25.0   (float64)  │
│ salário: 3500 (int)      │ ──────► │ salário: 3500.0 (float64)│
│ altura: 1.75 (float)     │         │ altura: 1.75  (float64)  │
│ peso: 72.3   (float)     │         │ peso: 72.3    (float64)  │
│ classe: "A"  (str)       │         │ classe: 0     (int64)    │ ← label encoding
└──────────────────────────┘         └──────────────────────────┘
                                      ↑
                                      Ponto de partida da quantização SIP-Q
                                      (SEMPRE float64 para features)
```

#### Como Funciona a Quantização: Por Grupo (Bit-Width Uniforme)

A quantização é aplicada **por grupo**, ou seja, num dado experimento **TODAS as features
recebem a mesma largura de bits**. Não quantizamos feature-a-feature com bits diferentes.

**Exemplo concreto com Wine dataset (13 features):**

```
Experimento com bit_width = 8 (int8):
┌─────────────────────────────────────────────────────────────────────────┐
│ Todas as 13 features são quantizadas para int8 (256 níveis cada)       │
│                                                                         │
│ Feature 0 (Alcohol):     float64 [11.0 ... 14.8] ──► int8 [0 ... 255] │
│ Feature 1 (Malic acid):  float64 [0.74 ... 5.80] ──► int8 [0 ... 255] │
│ Feature 2 (Ash):         float64 [1.36 ... 3.23] ──► int8 [0 ... 255] │
│ ...                                                                     │
│ Feature 12 (Proline):    float64 [278 ... 1680]   ──► int8 [0 ... 255] │
│                                                                         │
│ Cada feature tem o seu próprio scale e zero_point, MAS todas usam int8 │
└─────────────────────────────────────────────────────────────────────────┘
```

**O que muda entre features:** Cada feature tem o seu **próprio `scale` e `zero_point`**
(calculados a partir do `min` e `max` daquela coluna), porque os ranges são diferentes.
Mas o **tipo alvo é o mesmo** para todas (int8, int16 ou int32).

**O que é FIXO no experimento:** A largura de bits. Num experimento de 8-bit, tudo vai a 8-bit.
Depois corre-se outro experimento com 16-bit, e outro com 32-bit.

#### Fluxo Completo: Da Entrada ao Resultado

```
┌─────────────────────────────────────────────────────────────────┐
│ PASSO 1: Dados originais (float64)                              │
│                                                                  │
│ X_original = [[14.23, 1.71, 2.43, ...],   ← Wine, amostra 1   │
│               [13.20, 1.78, 2.14, ...],   ← Wine, amostra 2   │
│               ...]                                               │
│ Tipo: float64 | Memória: N × 13 × 8 bytes                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ PASSO 2: Quantização (float64 ──► int8)                        │
│                                                                  │
│ fit() no TREINO     — calibra scale/zero_point por coluna:      │
│   scale[i] = (max_train[i] - min_train[i]) / 255               │
│   zero_point[i] = min_train[i]                                  │
│                                                                  │
│ transform() no TREINO e no TESTE — mesmos params:               │
│   X_q[:, i] = round((X[:, i] - zero_point[i]) / scale[i])     │
│                                                                  │
│ X_quantized = [[255, 42, 118, ...],   ← inteiros 0-255        │
│                [210, 48,  87, ...],                              │
│                ...]                                              │
│ Tipo: uint8 | Memória: N × 13 × 1 byte  (8× menor!)           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ PASSO 3: Dequantização (int8 ──► float64 com ERRO)             │
│                                                                  │
│ X_reconstruido[:, i] = X_q[:, i] × scale[i] + zero_point[i]   │
│                                                                  │
│ X_reconstruido = [[14.22, 1.72, 2.44, ...],  ← QUASE igual    │
│                   [13.19, 1.79, 2.13, ...],  ← mas com ERRO   │
│                   ...]                                           │
│ Tipo: float64 | Erro: ||X_original - X_reconstruido||          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ PASSO 4: Usar X_reconstruido (float64 degradado) no modelo     │
│                                                                  │
│ quantizer.fit(X_train)          ← calibra APENAS no treino     │
│ X_r_train = quantizer.fit_transform(X_train)                    │
│ X_r_test  = quantizer.transform(X_test)   ← mesmos params!    │
│                                                                  │
│ model.fit(X_r_train, y_train)                                   │
│ predictions = model.predict(X_r_test)                           │
│                                                                  │
│ O modelo recebe float64, mas com a resolução de int8!           │
│ accuracy_quantized vs accuracy_baseline = DEGRADAÇÃO            │
└─────────────────────────────────────────────────────────────────┘
```

> **Nota Importante:** O scikit-learn NÃO aceita `int8` diretamente no `fit()`/`predict()`.
> Por isso, o ciclo é **float64 → int8 → float64** (roundtrip). O resultado é `float64`
> mas com os valores "degradados" — como se tivessem passado por um bottleneck de 8 bits.

#### Tabela de Conversão: Que Tipo Alvo Para Cada Bit-Width

| Bit-Width Config | Tipo Alvo (interno) | Níveis Discretos | Compressão | Erro Esperado |
|------------------|---------------------|-------------------|------------|---------------|
| **`bits=8`** | `uint8` | 256 | 8× | Alto — apenas 256 valores possíveis por feature |
| **`bits=16`** | `uint16` | 65.536 | 4× | Médio — resolução suficiente para a maioria dos dados |
| **`bits=16, fp16`** | `float16` | Contínuo (mantissa 10-bit) | 4× | Médio — erro proporcional à magnitude, sem calibração |
| **`bits=32`** | `uint32` | ~4 bilhões | 2× | Muito baixo — referência próxima do baseline |

> **Nota de escopo deste plano:** o protocolo core fixa `int8/int16/int32` (perfil `"integer"`).
> A trilha de extensão `float16` (perfil `"float16"`) opera apenas no ponto de 16 bits para
> comparação direta com `int16`. O ponto de 32-bit (`uint32`) funciona como referência
> quantizada de baixa degradação, aproximando-se do baseline `float64`.

#### E Se o Dado Original Já For Inteiro?

Mesmo que o dado "pareça inteiro" (ex: Digits tem pixels 0-16), no scikit-learn ele é
`float64`. A quantização trata-o da mesma forma:

```
Digits dataset — Pixel com valor 12 (parece inteiro, mas é float64):

float64:  12.000000000000000   (8 bytes, 64 bits de precisão)
     │
     ├──► int8:   12           (1 byte, valor exato neste caso!)
     ├──► int16:  12           (2 bytes, valor exato neste caso!)
    └──► int32:  12           (4 bytes, valor exato neste caso!)

Neste caso, a quantização NÃO introduz erro!
O erro aparece quando os valores são contínuos (ex: Wine: 14.23 → 14.22)
```

**Conclusão:** Dados que já são "inteiros disfarçados de float64" (como Digits, 0-16)
terão **menos degradação** com quantização, porque 256 níveis (int8) capturam perfeitamente
17 valores distintos. Dados verdadeiramente contínuos (como Wine, com 4+ casas decimais)
terão **mais degradação**.

#### Quantização de Parâmetros do Modelo: O Mesmo Princípio

A quantização de parâmetros segue **exactamente o mesmo processo**, mas aplicado aos
pesos internos do modelo em vez dos dados:

```
QUANTIZAÇÃO DE DADOS:             QUANTIZAÇÃO DE PARÂMETROS:
X_original (float64)               model.coefs_ (float64)
      │                                  │
      ▼                                  ▼
X_quantized (int8)                 coefs_quantized (int8)
      │                                  │
      ▼                                  ▼
X_reconstruido (float64 c/ erro)   coefs_reconstruido (float64 c/ erro)
      │                                  │
      ▼                                  ▼
model.fit(X_reconstruido)          model_copy.coefs_ = coefs_reconstruido
model.predict(X_test)              model_copy.predict(X_test)
```

**Diferença fundamental:**
- **Quantizar dados:** Degrada a qualidade dos INPUTS — o modelo "vê" dados menos precisos
- **Quantizar parâmetros:** Degrada o CONHECIMENTO aprendido — o modelo "sabe" menos
- **Híbrido:** Ambos degradados — simula implantação real em dispositivo de baixa precisão

#### Resumo Visual: Protocolo Experimental Completo

```
Para cada (dataset, algoritmo, bit_width, seed):
═══════════════════════════════════════════════════════════════════

  ┌── BASELINE ──────────────────────────────────────────────────┐
  │ X (float64) ──► model.fit() ──► model.predict() ──► acc_base│
  └──────────────────────────────────────────────────────────────┘

  ┌── MODO 1: data_only ─────────────────────────────────────────┐
  │ X (float64) ──► quantize ──► dequantize ──► X' (float64     │
  │                                               com erro)      │
  │ X' ──► model.fit() ──► model.predict() ──► acc_data_quant   │
  └──────────────────────────────────────────────────────────────┘

  ┌── MODO 2: model_only ────────────────────────────────────────┐
  │ X (float64) ──► model.fit() ──► quantize pesos ──►          │
  │                                  dequantize pesos ──►        │
  │                                  model'.predict() ──►        │
  │                                  acc_model_quant             │
  └──────────────────────────────────────────────────────────────┘

  ┌── MODO 3: hybrid ────────────────────────────────────────────┐
  │ X' (dados quantizados) ──► model.fit() ──►                   │
  │                             quantize pesos ──►               │
  │                             model'.predict(X'_test) ──►      │
  │                             acc_hybrid                       │
  └──────────────────────────────────────────────────────────────┘

  Resultado: acc_base vs acc_data vs acc_model vs acc_hybrid
═══════════════════════════════════════════════════════════════════
```

### Arquitetura de Alto Nível

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Módulo SIP-Q Quantização                       │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  QuantizationConfig                                         │   │
│  │  • data_bits: Tuple[BitWidth]  → (8, 16, 32)              │   │
│  │  • model_bits: Tuple[BitWidth] → (8, 16, 32)              │   │
│  │  • dtype_profile: DTypeProfile → "integer"|"float16"      │   │
│  │  • method: QuantMethod     → uniform|minmax|kmeans|percentile│  │
│  │  • enable_hybrid: bool     → True                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
  ┌───────────▼───────────┐         ┌─────────────▼─────────────┐
  │  Quantização de Dados │         │  Quantização de Modelos   │
  │                       │         │                           │
  │  Entrada: float64     │         │  Entrada: modelo treinado │
  │  Saída: uint8/uint16  │         │  Saída: pesos degradados  │
  │         /uint32/fp16  │         │   (roundtrip float64)     │
  │                       │         │                           │
  │  • FeatureQuantizer   │         │  • WeightQuantizer        │
  │  • 4 métodos (int)    │         │  • 6 algoritmos suportados│
  │  • cast direto (fp16) │         │  • AC: sem pesos → noop   │
  └───────────┬───────────┘         └─────────────┬─────────────┘
              │                                   │
              └─────────────┬─────────────────────┘
                            │
              ┌─────────────▼─────────────────┐
              │  Quantização Híbrida          │
              │  (Dados int8 + Modelo int16)  │
              │  Medir degradação acumulada   │
              └─────────────┬─────────────────┘
                            │
              ┌─────────────▼─────────────────┐
              │  Motor de Avaliação           │
              │                               │
              │  • accuracy_degradation       │
              │  • memory_reduction (bytes)   │
              │  • inference_overhead_pct     │
              │  • quantization_error (MSE)   │
              │  • silhouette_degradation     │ ← para clustering
              └───────────────────────────────┘
```

### Estrutura de Módulos

```
smart_inference_ai_fusion/
│
├── quantization/                              ← NOVO MÓDULO
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                          # QuantizationConfig (dataclass)
│   │   ├── types.py                           # QuantMethod, BitWidth (Literal types)
│   │   ├── methods.py                         # uniform, minmax, kmeans, percentile
│   │   └── result.py                          # QuantizationResult (Pydantic)
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── feature_quantizer.py               # FeatureQuantizer (float64 → int8/16)
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   └── weight_quantizer.py                # WeightQuantizer (pesos → int8/16)
│   │
│   ├── hybrid/
│   │   ├── __init__.py
│   │   └── hybrid_quantizer.py                # Combina FeatureQuantizer + WeightQuantizer
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py                         # compute_supervised_metrics, compute_clustering_metrics
│       └── benchmark.py                       # benchmark_time, benchmark_memory
│
├── experiments/
│   └── quantization_experiment.py             # QuantizationExperiment
│
├── scripts/
│   └── case4.py                               ← NOVO ESTUDO DE CASO
│
└── tests/
    └── test_quantization/
        ├── test_config.py
        ├── test_methods.py
        ├── test_feature_quantizer.py
        ├── test_weight_quantizer.py
        └── test_evaluation.py
```

### Fluxo de Dados

```
┌─────────────────────┐
│   Dataset Original  │
│   (float64)         │
└──────────┬──────────┘
           │
           ├─────────────────┬──────────────────┐
           │                 │                  │
           ▼                 ▼                  ▼
    ┌──────────┐     ┌──────────┐      ┌──────────┐
    │ Baseline │     │ Quant.   │      │ Quant.   │
    │ (Normal) │     │ Dados    │      │ Modelo   │
    └──────────┘     └──────────┘      └──────────┘
           │                 │                  │
           │                 │                  │
           ├─────────────────┴──────────────────┤
           │                                    │
           ▼                                    ▼
    ┌──────────┐                        ┌──────────┐
    │ Avaliação│◄───────────────────────│ Quant.   │
    │ Baseline │                        │ Híbrida  │
    └──────────┘                        └──────────┘
           │                                    │
           └────────────┬───────────────────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │  Comparação  │
                 │  e Análise   │
                 └──────────────┘
```

### Comparação de Métodos de Quantização

| Método | Tipo | Complexidade | Erro Típico | Uso Recomendado |
|--------|------|-------------|-------------|-----------------|
| **Uniform** | Simétrico | O(n) | Médio | Dados normalizados ou de distribuição uniforme |
| **Min-Max** | Assimétrico | O(n) | Baixo | Dados com ranges variados entre features |
| **K-Means** | Não-uniforme | O(n×k×i) | Muito baixo | Distribuições complexas ou multimodais |
| **Percentile** | Não-uniforme | O(n log n) | Baixo | Dados com outliers significativos |

### Larguras de Bits Suportadas

| Perfil | Bits | Representação | Precisão | Uso Típico |
|------|------|----------------|----------|------------|
| **int** | 8 | `int8` | Baixa | Edge devices, máxima compressão |
| **int** | 16 | `int16` | Média | Balanço acurácia/eficiência |
| **float** | 16 | `float16` | Média (alcance maior, mantissa menor) | GPUs/NPUs com half-precision |
| **int** | 32 | `int32` | Alta | Referência quantizada próxima ao baseline |

---

## 🚀 Fases de Implementação

### Visão Geral das Fases

| Fase | Nome | Duração | Descrição |
|------|------|---------|-----------|
| **0** | Preparação | 2 dias | Setup de ambiente, estrutura de diretórios, branch |
| **1** | Fundação | 3 dias | Classes base: `QuantizationConfig`, `QuantizationResult`, tipos |
| **2** | Quantização de Dados | 5 dias | 4 métodos de quantização de features (`float64` para `int8`/`int16`/`float16`/`int32`) |
| **3** | Quantização de Modelos | 5 dias | Quantizar pesos internos dos 6 algoritmos selecionados |
| **4** | Avaliação e Métricas | 3 dias | Métricas supervisionadas e de clustering, benchmark |
| **5** | Integração | 5 dias | `case4.py`, integração com SIP/SIP-V, CI |
| **6** | Análise e Documentação | 5 dias | Resultados, visualizações, relatório técnico |

**Total estimado: ~4 semanas**

---

### Fase 0 - Preparação e Setup (Dias 1-2)

**Objetivo:** Preparar o ambiente de trabalho para o desenvolvimento do módulo SIP-Q.
Nesta fase não se escreve lógica de quantização - apenas se cria a infraestrutura
necessária para que as fases seguintes possam começar sem impedimentos.

#### Tarefas:

1. **Criar branch de desenvolvimento**
   ```bash
   git checkout -b feature/sip-q-quantization
   ```

2. **Criar estrutura de diretórios**
   - Criar todos os diretórios e ficheiros `__init__.py` da árvore do módulo `quantization/`
   - Garantir que o import `from smart_inference_ai_fusion.quantization import ...` funciona

3. **Verificar dependências**
   - O módulo usa exclusivamente `numpy`, `scikit-learn`, `pandas` e `pydantic` (já instalados)
   - Não são necessárias dependências novas no `pyproject.toml`

4. **Configurar testes**
   - Criar diretorio `tests/test_quantization/`
   - Verificar que `pytest` descobre os novos testes

#### Entregáveis:
- [ ] Branch `feature/sip-q-quantization` criada
- [ ] Estrutura de diretórios completa com `__init__.py`
- [ ] `pytest` executa sem erros (0 testes, 0 falhas)

#### Validação:
```bash
python -c "import smart_inference_ai_fusion.quantization; print('OK')"
pytest tests/test_quantization/ -v  # Deve executar sem erros
```

---

### Fase 1 - Fundação: Classes Base e Tipos (Dias 3-5)

**Objetivo:** Implementar as estruturas de dados fundamentais que todo o módulo SIP-Q
vai utilizar. Estas classes definem a configuração dos experimentos, os tipos de dados
envolvidos é o formato dos resultados.

#### 1.1 Definicao de Tipos (`core/types.py`)

O módulo precisa de tipos bem definidos para evitar erros de configuração:

```python
"""Tipos base para quantização."""
from typing import Literal

# Métodos de quantização suportados
QuantMethod = Literal["uniform", "minmax", "kmeans", "percentile"]

# Larguras de bits suportadas (core)
BitWidth = Literal[8, 16, 32]

# Perfil de tipo numérico para o ponto de 16 bits
# "integer" → int8/int16/int32 (core — escala + zero_point)
# "float16" → float16 no ponto de 16 bits (extensão — cast direto)
DTypeProfile = Literal["integer", "float16"]

# Tipos numpy alvo (core)
# 8 bits  -> np.uint8   (256 níveis, 1 byte)
# 16 bits -> np.uint16  (65536 níveis, 2 bytes)
# 32 bits -> np.uint32  (~4 bilhões de níveis, 4 bytes)
# 16 bits (float) -> np.float16 (cast direto, 2 bytes)
# Baseline: np.float64 (precisão de ~16 dígitos, 8 bytes)
```

#### 1.2 Configuração (`core/config.py`)

Classe imutável que define todos os parâmetros de um experimento de quantização:

```python
"""Configuração de experimentos de quantização."""
from dataclasses import dataclass, field
from typing import List
from .types import QuantMethod, BitWidth, DTypeProfile

@dataclass(frozen=True)
class QuantizationConfig:
    """Configuração imutável para um experimento SIP-Q."""

    # Que larguras de bits testar?
    data_bits: tuple[BitWidth, ...] = (8, 16, 32)
    model_bits: tuple[BitWidth, ...] = (8, 16, 32)

    # Perfil de tipo numérico para quantização
    # "integer": int8/int16/int32 (quantização com escala+zero_point)
    # "float16": usa float16 no ponto de 16 bits (cast direto)
    dtype_profile: DTypeProfile = "integer"

    # Que método de quantização usar? (aplica-se apenas ao perfil "integer")
    method: QuantMethod = "uniform"

    # Testar combinação dados + modelo?
    enable_hybrid: bool = True

    # Quantas amostras para calibração (k-means/percentile)?
    calibration_samples: int = 1000

    # Seed para reprodutibilidade
    random_seed: int = 42

    def __post_init__(self):
        if self.dtype_profile == "float16" and self.method != "uniform":
            import warnings
            warnings.warn(
                "float16 usa cast direto — o campo 'method' é ignorado neste perfil."
            )
```

#### 1.3 Schema de Resultado (`core/result.py`)

Formato padronizado para guardar resultados - compatível com os JSON dos cases anteriores:

```python
"""Resultado de um experimento de quantização."""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class QuantizationResult(BaseModel):
    """Resultado padronizado de um experimento SIP-Q."""

    # Identificação
    experiment_type: str          # "data_quant", "model_quant", "hybrid"
    dataset_name: str
    algorithm_name: str
    quantization_method: str
    bit_width: int
    dtype_profile: str = "integer"  # "integer" ou "float16"

    # Métricas - Supervisionado
    baseline_accuracy: Optional[float] = None
    quantized_accuracy: Optional[float] = None
    accuracy_degradation: Optional[float] = None  # baseline - quantized

    # Métricas - Clustering
    baseline_silhouette: Optional[float] = None
    quantized_silhouette: Optional[float] = None
    silhouette_degradation: Optional[float] = None

    # Eficiencia
    baseline_memory_bytes: int
    quantized_memory_bytes: int
    compression_ratio: float      # ex: 8.0 para float64->int8

    baseline_time_ms: float
    quantized_time_ms: float      # inclui tempo de quant + dequant + inferência
    overhead_pct: float            # (quantized_time - baseline_time) / baseline_time * 100

    # Erro de quantização
    quantization_mse: float       # MSE entre original e dequantizado

    # Metadados
    seed: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

#### 1.4 Testes da Fase 1

```python
# tests/test_quantization/test_config.py
def test_config_defaults():
    config = QuantizationConfig()
    assert config.data_bits == (8, 16, 32)
    assert config.method == "uniform"

def test_config_immutable():
    config = QuantizationConfig()
    with pytest.raises(FrozenInstanceError):
        config.method = "kmeans"

def test_result_serialization():
    result = QuantizationResult(
        experiment_type="data_quant",
        dataset_name="Wine", algorithm_name="KNN",
        quantization_method="uniform", bit_width=8,
        baseline_accuracy=0.95, quantized_accuracy=0.90,
        accuracy_degradation=0.05,
        baseline_memory_bytes=11392, quantized_memory_bytes=1424,
        compression_ratio=8.0,
        baseline_time_ms=1.2, quantized_time_ms=0.8,
        overhead_pct=5.2, quantization_mse=0.003, seed=42
    )
    json_str = result.model_dump_json()
    assert "Wine" in json_str
```

#### Entregáveis:
- [ ] `QuantizationConfig` com validação de tipos
- [ ] `QuantizationResult` serializavel para JSON (Pydantic)
- [ ] Tipos `QuantMethod` e `BitWidth` definidos
- [ ] >= 10 testes unitários para classes base

#### Validação:
```bash
pytest tests/test_quantization/test_config.py -v
pylint smart_inference_ai_fusion/quantization/core/ --fail-under=9.5
```

---

### Fase 2 - Quantização de Dados (Dias 6-10)

**Objetivo:** Implementar os 4 métodos de quantização que convertem features de `float64`
para tipos de menor precisão (`int8`, `int16`, `int32`). Esta é a fase central do SIP-Q
- e aqui que se implementa a transformacao que degrada a resolução dos dados de entrada.

Cada método aborda o problema de forma diferente:
- **Uniforme**: distribui os valores igualmente pelo range disponível
- **Min-Max**: normaliza para [0,1] e depois quantiza - adapta-se ao range dos dados
- **K-Means**: agrupa valores similares - adapta-se a distribuição dos dados
- **Percentile**: usa percentis para definir fronteiras - robusto a outliers

#### 2.1 Métodos de Quantização (`core/methods.py`)

Cada método recebe um array `float64` e devolve o array quantizado mais os parâmetros
necessarios para a operação inversa (dequantização):

```python
"""
Métodos de quantização para SIP-Q.

Todos os métodos seguem a mesma interface:
  Entrada:  np.ndarray (float64)
  Saida:    (np.ndarray quantizado, parâmetros para dequantização)
"""
import numpy as np
from typing import Tuple, Dict, Any

def uniform_quantize(data: np.ndarray, num_bits: int = 8
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Quantização uniforme simétrica.

    Mapeia o range [min, max] para [0, 2^bits - 1] de forma linear.
    E o método mais simples e rápido, mas ignora a distribuição dos dados.

    Conversão de tipos:
      float64 -> uint8  (8 bits, 256 níveis)
      float64 -> uint16 (16 bits, 65536 níveis)
      float64 -> uint32 (32 bits, ~4B níveis)
    """
    min_val, max_val = data.min(), data.max()
    if min_val == max_val:
        return np.zeros_like(data, dtype=np.uint8), {
            "scale": 1.0, "zero_point": 0.0, "min": min_val, "max": max_val,
            "method": "uniform"
        }

    qmax = (2 ** num_bits) - 1
    scale = (max_val - min_val) / qmax
    zero_point = min_val

    quantized = np.clip(np.round((data - zero_point) / scale), 0, qmax)

    # Nota: usamos tipos unsigned para evitar overflow.
    # int8 signed vai até 127, mas qmax=255 para 8 bits.
    dtype_map = {8: np.uint8, 16: np.uint16, 32: np.uint32}
    quantized = quantized.astype(dtype_map.get(num_bits, np.uint16))

    return quantized, {
        "scale": scale, "zero_point": zero_point, "method": "uniform"
    }


def dequantize(quantized: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Operação inversa: quantizado -> float64 (com erro de reconstrução).
    Despacha automaticamente para o método correto com base nos params.
    """
    method = params.get("method", "uniform")

    if method == "kmeans":
        # Cada label mapeia para o centroide correspondente
        centroids = params["centroids"]
        return centroids[quantized.astype(int)]

    if method == "percentile":
        # Cada bin mapeia para o ponto médio do intervalo
        bins = params["bins"]
        bin_centers = (bins[:-1] + bins[1:]) / 2
        indices = np.clip(quantized.astype(int), 0, len(bin_centers) - 1)
        return bin_centers[indices]

    if method == "minmax":
        # Reconstruir para [0,1] e depois para range original
        reconstructed_01 = params["scale"] * quantized.astype(np.float64) + params["zero_point"]
        original_min = params["original_min"]
        original_max = params["original_max"]
        return reconstructed_01 * (original_max - original_min) + original_min

    # uniform (default)
    return params["scale"] * quantized.astype(np.float64) + params["zero_point"]


def minmax_quantize(data: np.ndarray, num_bits: int = 8
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normaliza para [0, 1] antes de quantizar.
    Melhor que uniforme quando features tem ranges muito diferentes.
    """
    min_val, max_val = data.min(), data.max()
    if min_val == max_val:
        return np.zeros_like(data, dtype=np.uint8), {
            "min": min_val, "max": max_val, "method": "minmax"
        }
    normalized = (data - min_val) / (max_val - min_val)
    quantized, params = uniform_quantize(normalized, num_bits)
    params["method"] = "minmax"
    params["original_min"] = min_val
    params["original_max"] = max_val
    return quantized, params


def kmeans_quantize(data: np.ndarray, num_bits: int = 8
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Quantização não-uniforme via k-means.
    Agrupa valores semelhantes - bom para distribuições complexas.
    Mais lento que uniforme (requer treino do k-means).

    num_clusters = 2^num_bits (ex: 256 para 8 bits)
    """
    from sklearn.cluster import MiniBatchKMeans

    n_clusters = min(2 ** num_bits, len(np.unique(data.flatten())))
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3)
    flat = data.flatten().reshape(-1, 1)
    labels = kmeans.fit_predict(flat)

    dtype_map = {8: np.uint8, 16: np.uint16, 32: np.uint32}
    target_dtype = dtype_map.get(num_bits, np.uint16)
    return labels.reshape(data.shape).astype(target_dtype), {
        "centroids": kmeans.cluster_centers_.flatten(),
        "method": "kmeans"
    }


def percentile_quantize(data: np.ndarray, num_bits: int = 8
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Quantização baseada em percentis.
    Define fronteiras pela distribuição dos dados - robusto a outliers.
    """
    num_bins = 2 ** num_bits
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(data.flatten(), percentiles)
    bins = np.unique(bins)  # Remover duplicados

    quantized = np.digitize(data.flatten(), bins) - 1
    quantized = np.clip(quantized, 0, len(bins) - 2)

    dtype_map = {8: np.uint8, 16: np.uint16, 32: np.uint32}
    target_dtype = dtype_map.get(num_bits, np.uint16)
    return quantized.reshape(data.shape).astype(target_dtype), {
        "bins": bins, "method": "percentile"
    }
```

#### 2.2 Feature Quantizer (`data/feature_quantizer.py`)

Classe de alto nivel que aplica qualquer método de quantização feature-a-feature:

```python
"""Quantizador de features - aplica quantização coluna a coluna."""
import numpy as np
from typing import Dict, List, Any
from ..core.methods import (
    uniform_quantize, minmax_quantize,
    kmeans_quantize, percentile_quantize, dequantize
)

METHODS = {
    "uniform": uniform_quantize,
    "minmax": minmax_quantize,
    "kmeans": kmeans_quantize,
    "percentile": percentile_quantize,
}

class FeatureQuantizer:
    """
    Quantiza features de um dataset coluna a coluna.

    Fluxo (perfil integer):
      X_train -> fit(X_train)           → calibra scale/zero_point por coluna
      X       -> transform(X)           → quantiza usando params calibrados
      X_q     -> inverse_transform(X_q) → reconstrói float64 com erro

    Fluxo (perfil float16):
      X -> transform(X)                 → cast direto (sem calibração)
      X_q -> inverse_transform(X_q)     → cast inverso

    IMPORTANTE: calibrar (fit) apenas no conjunto de treino para evitar
    vazamento de informação do conjunto de teste.
    """

    DTYPE_MAP = {8: np.uint8, 16: np.uint16, 32: np.uint32}

    def __init__(self, method: str = "uniform", num_bits: int = 8,
                 dtype_profile: str = "integer"):
        self.method = method
        self.num_bits = num_bits
        self.dtype_profile = dtype_profile
        self._params: List[Dict[str, Any]] = []
        self._fitted = False

    def fit(self, X: np.ndarray) -> "FeatureQuantizer":
        """Calibrar parâmetros de quantização (apenas no treino)."""
        if self.dtype_profile == "float16":
            self._params = [{"method": "float16_cast"}] * X.shape[1]
            self._fitted = True
            return self

        quantize_fn = METHODS[self.method]
        self._params = []
        for i in range(X.shape[1]):
            _, params = quantize_fn(X[:, i], self.num_bits)
            self._params.append(params)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Quantizar usando parâmetros já calibrados."""
        assert self._fitted, "Chamar fit() antes de transform()."

        if self.dtype_profile == "float16":
            return X.astype(np.float16)

        target_dtype = self.DTYPE_MAP.get(self.num_bits, np.uint16)
        X_quantized = np.zeros(X.shape, dtype=target_dtype)
        quantize_fn = METHODS[self.method]
        for i in range(X.shape[1]):
            col_q, _ = quantize_fn(X[:, i], self.num_bits)
            X_quantized[:, i] = col_q
        return X_quantized

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Conveniência: fit + transform no mesmo array."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_quantized: np.ndarray) -> np.ndarray:
        """Dequantizar - reconstruir float64 a partir do quantizado."""
        if self.dtype_profile == "float16":
            return X_quantized.astype(np.float64)

        X_reconstructed = np.zeros(X_quantized.shape, dtype=np.float64)
        for i, params in enumerate(self._params):
            X_reconstructed[:, i] = dequantize(X_quantized[:, i], params)
        return X_reconstructed
```

#### 2.3 Testes da Fase 2

```python
# tests/test_quantization/test_methods.py
def test_uniform_int8_range():
    """Valores quantizados 8-bit devem estar em [0, 255]."""
    X = np.random.randn(100)
    q, params = uniform_quantize(X, num_bits=8)
    assert q.dtype == np.uint8
    assert q.min() >= 0 and q.max() <= 255

def test_dequantize_error():
    """Erro de reconstrução 16-bit deve ser menor que 8-bit."""
    X = np.random.randn(1000)
    q8, p8 = uniform_quantize(X, 8)
    q16, p16 = uniform_quantize(X, 16)
    err8 = np.mean((X - dequantize(q8, p8)) ** 2)
    err16 = np.mean((X - dequantize(q16, p16)) ** 2)
    assert err16 < err8  # Mais bits = menos erro

def test_feature_quantizer_iris():
    """Testar pipeline completo com Iris."""
    from sklearn.datasets import load_iris
    X, _ = load_iris(return_X_y=True)
    fq = FeatureQuantizer(method="uniform", num_bits=16)
    X_q = fq.fit_transform(X)
    X_r = fq.inverse_transform(X_q)
    mse = np.mean((X - X_r) ** 2)
    assert mse < 0.01  # Erro aceitável para 16-bit
```

#### Entregáveis:
- [ ] 4 funcoes de quantização (`uniform`, `minmax`, `kmeans`, `percentile`)
- [ ] Função `dequantize()` para operação inversa
- [ ] `FeatureQuantizer` com `fit_transform()` / `inverse_transform()`
- [ ] >= 15 testes unitários para métodos e quantizador
- [ ] Benchmark: tabela de erro (MSE) por método x bits

#### Validação:
```bash
pytest tests/test_quantization/test_methods.py tests/test_quantization/test_feature_quantizer.py -v
```

---

### Fase 3 - Quantização de Modelos (Dias 11-15)

**Objetivo:** Implementar a quantização de pesos internos de modelos já treinados.
Diferente da Fase 2 (que quantiza os dados de entrada), está fase quantiza os parâmetros
aprendidos pelo modelo (coeficientes, pesos, centróides).

O impacto varia por tipo de modelo:
- **KNN**: não tem pesos tradicionais, mas armazena todos os dados de treino - quantizar
  `_fit_X` reduz drasticamente a memória
- **Decision Tree**: tem limiares de split (`threshold`) - quantizar pode alterar decisões
- **MLP**: tem muitos pesos (`coefs_`, `intercepts_`) - alto impacto da quantização
- **MiniBatchKMeans**: centróides (`cluster_centers_`) - afeta atribuição de clusters
- **GMM**: médias e covariâncias - muito sensível a precisão
- **Agglomerative**: recalcula distâncias - afetado indiretamente pela quantização de X

#### 3.1 Weight Quantizer (`model/weight_quantizer.py`)

```python
"""
Quantizador de pesos de modelos scikit-learn.

Para cada tipo de modelo, identifica os atributos internos relevantes
e aplica quantização + dequantização (mantendo o tipo float64 para
compatibilidade com predict(), mas com valores degradados).
"""
import numpy as np
import copy
from sklearn.base import BaseEstimator
from ..core.methods import uniform_quantize, dequantize

class WeightQuantizer:
    """
    Quantiza pesos de modelos treinados.

    O modelo continua a operar em float64 internamente, mas os valores
    dos pesos passam pelo ciclo quantize->dequantize, introduzindo
    o erro de quantização nos parâmetros.

    Isto simula o comportamento de um modelo exportado para hardware
    de menor precisão.
    """

    def __init__(self, num_bits: int = 8, method: str = "uniform",
                 dtype_profile: str = "integer"):
        self.num_bits = num_bits
        self.method = method
        self.dtype_profile = dtype_profile

    def quantize_model(self, model):
        """
        Devolve uma cópia do modelo com pesos quantizados.

        Aceita:
          - Framework BaseModel (acede a model.model para chegar ao estimador sklearn)
          - sklearn BaseEstimator directamente

        Devolve o mesmo tipo recebido (BaseModel ou BaseEstimator).
        O modelo original NÃO é alterado.
        """
        estimator = getattr(model, "model", model)  # BaseModel wrapper → sklearn
        estimator_copy = copy.deepcopy(estimator)

        # Modelos supervisionados
        if hasattr(estimator_copy, 'coefs_'):
            # MLP - Lista de arrays de pesos por camada
            estimator_copy.coefs_ = [self._quantize_roundtrip(w) for w in estimator_copy.coefs_]
            estimator_copy.intercepts_ = [self._quantize_roundtrip(b) for b in estimator_copy.intercepts_]

        elif hasattr(estimator_copy, 'tree_'):
            # Decision Tree - Limiares de split
            estimator_copy.tree_.threshold[:] = self._quantize_roundtrip(
                estimator_copy.tree_.threshold
            )

        elif hasattr(estimator_copy, '_fit_X'):
            # KNN - Dados de treino armazenados
            estimator_copy._fit_X = self._quantize_roundtrip(estimator_copy._fit_X)

        # Modelos não-supervisionados
        if hasattr(estimator_copy, 'cluster_centers_'):
            # MiniBatchKMeans - Centróides
            estimator_copy.cluster_centers_ = self._quantize_roundtrip(
                estimator_copy.cluster_centers_
            )

        if hasattr(estimator_copy, 'means_'):
            # GMM - Médias e covariâncias
            estimator_copy.means_ = self._quantize_roundtrip(estimator_copy.means_)
            if hasattr(estimator_copy, 'covariances_'):
                estimator_copy.covariances_ = self._quantize_roundtrip(
                    estimator_copy.covariances_
                )

        # Nota: AgglomerativeClustering não armazena pesos internos —
        # em mode "model_only", o resultado será igual ao baseline.

        # Re-atribuir ao wrapper se aplicável
        if hasattr(model, "model"):
            model_out = copy.deepcopy(model)
            model_out.model = estimator_copy
            return model_out
        return estimator_copy

    def _quantize_roundtrip(self, arr: np.ndarray) -> np.ndarray:
        """
        Ciclo completo: float64 -> intN/fp16 -> float64.
        O resultado é float64 mas com a resolução reduzida.
        """
        # float16: cast direto (sem escala)
        if self.dtype_profile == "float16":
            return arr.astype(np.float16).astype(np.float64)

        # integer: quantização com escala (respeita self.method)
        quantize_fn = METHODS[self.method]
        original_shape = arr.shape
        flat = arr.flatten()
        quantized, params = quantize_fn(flat, self.num_bits)
        reconstructed = dequantize(quantized, params)
        return reconstructed.reshape(original_shape)
```

#### 3.2 Exemplos de Impacto por Modelo

O impacto da quantização difere fundamentalmente entre tipos de modelo:

| Modelo | O que e quantizado | 8-bit esperado | 16-bit esperado |
|--------|-------------------|----------------|-----------------|
| **KNN** | `_fit_X` (NxD floats) | Alto impacto - distâncias distorcidas | Impacto baixo |
| **Decision Tree** | `threshold` (~nos floats) | Médio - pode alterar decisões | Mínimo |
| **MLP** | `coefs_` + `intercepts_` | Alto - muitos parâmetros | Moderado |
| **MiniBatchKMeans** | `cluster_centers_` | Médio - centróides deslocados | Mínimo |
| **GMM** | `means_` + `covariances_` | Alto - covariâncias sensiveis | Moderado |
| **Agglomerative** | Sem pesos (recalcula) | Sem efeito direto | Sem efeito |

#### 3.3 Testes da Fase 3

```python
# tests/test_quantization/test_weight_quantizer.py
def test_knn_quantization():
    """KNN quantizado deve manter acurácia razoável em 16-bit."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.datasets import load_wine
    X, y = load_wine(return_X_y=True)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X[:120], y[:120])
    acc_baseline = model.score(X[120:], y[120:])

    wq = WeightQuantizer(num_bits=16)
    model_q = wq.quantize_model(model)
    acc_quant = model_q.score(X[120:], y[120:])

    assert abs(acc_baseline - acc_quant) < 0.10  # Max 10% degradação

def test_kmeans_quantization():
    """Centróides quantizados devem manter silhouette razoável."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=300, centers=3, random_state=42)
    model = MiniBatchKMeans(n_clusters=3, random_state=42)
    model.fit(X)
    sil_baseline = silhouette_score(X, model.predict(X))

    wq = WeightQuantizer(num_bits=16)
    model_q = wq.quantize_model(model)
    sil_quant = silhouette_score(X, model_q.predict(X))

    assert sil_quant > sil_baseline * 0.8  # Max 20% degradação
```

#### Entregáveis:
- [ ] `WeightQuantizer` com suporte para os 6 algoritmos
- [ ] Copia profunda (modelo original inalterado)
- [ ] Testes por tipo de modelo
- [ ] Tabela de impacto empírico (8-bit vs 16-bit)

---

### Fase 4 - Avaliação e Métricas (Dias 16-18)

**Objetivo:** Implementar o sistema de avaliação que mede o impacto da quantização.
Como temos algoritmos supervisionados e não-supervisionados, precisamos de dois
conjuntos de métricas distintos.

#### 4.1 Métricas Supervisionadas vs. Nao-Supervisionadas

| Categoria | Metrica | Formula | Aplica-se a |
|-----------|---------|---------|-------------|
| **Acurácia** | accuracy_degradation | baseline_acc - quantized_acc | Supervisionado |
| **Acurácia** | f1_degradation | baseline_f1 - quantized_f1 | Supervisionado |
| **Clustering** | silhouette_degradation | baseline_sil - quantized_sil | Nao-supervisionado |
| **Clustering** | ari_degradation | baseline_ARI - quantized_ARI | Nao-supervisionado |
| **Eficiencia** | memory_reduction | 1 - (quant_bytes / base_bytes) | Ambos |
| **Eficiencia** | compression_ratio | 64 / num_bits | Ambos |
| **Eficiencia** | overhead_pct | (quant_time − base_time) / base_time × 100 | Ambos |
| **Erro** | quantization_mse | mean((X - X_reconstructed)^2) | Ambos |

#### 4.2 Implementação (`evaluation/metrics.py`)

```python
"""
Métricas de avaliação para quantização.

Duas funcoes principais:
  - compute_supervised_metrics(): accuracy, f1, precision, recall
  - compute_clustering_metrics(): silhouette, ARI, NMI
"""
import numpy as np
import time
from typing import Dict, Any, Callable
from sklearn.metrics import (
    accuracy_score, f1_score,
    silhouette_score, adjusted_rand_score,
    normalized_mutual_info_score
)

def compute_supervised_metrics(
    y_true: np.ndarray,
    y_pred_baseline: np.ndarray,
    y_pred_quantized: np.ndarray
) -> Dict[str, float]:
    """Métricas para modelos supervisionados (classificação)."""
    acc_base = accuracy_score(y_true, y_pred_baseline)
    acc_quant = accuracy_score(y_true, y_pred_quantized)
    f1_base = f1_score(y_true, y_pred_baseline, average="weighted")
    f1_quant = f1_score(y_true, y_pred_quantized, average="weighted")

    return {
        "baseline_accuracy": acc_base,
        "quantized_accuracy": acc_quant,
        "accuracy_degradation": acc_base - acc_quant,
        "baseline_f1": f1_base,
        "quantized_f1": f1_quant,
        "f1_degradation": f1_base - f1_quant,
    }


def compute_clustering_metrics(
    X: np.ndarray,
    labels_baseline: np.ndarray,
    labels_quantized: np.ndarray,
    labels_true: np.ndarray = None
) -> Dict[str, float]:
    """Métricas para modelos não-supervisionados (clustering)."""
    sil_base = silhouette_score(X, labels_baseline)
    sil_quant = silhouette_score(X, labels_quantized)

    result = {
        "baseline_silhouette": sil_base,
        "quantized_silhouette": sil_quant,
        "silhouette_degradation": sil_base - sil_quant,
    }

    if labels_true is not None:
        ari_base = adjusted_rand_score(labels_true, labels_baseline)
        ari_quant = adjusted_rand_score(labels_true, labels_quantized)
        result["baseline_ari"] = ari_base
        result["quantized_ari"] = ari_quant
        result["ari_degradation"] = ari_base - ari_quant

    return result


def benchmark_inference(
    predict_fn: Callable,
    X: np.ndarray,
    n_runs: int = 50
) -> float:
    """Medir tempo médio de inferencia em milissegundos."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        predict_fn(X)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return float(np.median(times))
```

#### 4.3 Pipeline de Avaliação Completo

```
+-----------------------------------+
|  1. Baseline (float64)            |
|  - Treinar modelo                 |
|  - Medir accuracy/silhouette      |
|  - Medir tempo e memória          |
+------------------+----------------+
                   |
    +--------------+--------------+
    v                             v
+----------+               +----------+
| 2. Quant.|               | 3. Quant.|
| de Dados |               | de Modelo|
| (8,16,32)|               | (8,16,32)|
+----------+               +----------+
    |                             |
    +--------------+--------------+
                   v
           +--------------+
           | 4. Híbrido   |
           | (dados+model)|
           +------+-------+
                  v
           +--------------+
           | 5. Comparação|
           | - Tabelas    |
           | - Gráficos   |
           | - Pareto     |
           +--------------+
```

#### Entregáveis:
- [ ] `compute_supervised_metrics()` para classificação
- [ ] `compute_clustering_metrics()` para clustering
- [ ] `benchmark_inference()` para tempos
- [ ] >= 8 testes para métricas

---

### Fase 5 - Integração e Case Study 4 (Dias 19-23)

**Objetivo:** Criar o `case4.py` que executa os experimentos SIP-Q de forma completa,
integrando quantização de dados, modelos e avaliação. Este script segue o padrão dos
cases anteriores (case1, case2, case3) e produz resultados em JSON.

#### 5.1 Classe de Experimento

```python
"""Experimento de quantização SIP-Q."""
from typing import List, Optional, Type, Union
import numpy as np
from smart_inference_ai_fusion.core.base_model import BaseModel
from smart_inference_ai_fusion.utils.types import DatasetSourceType, SklearnDatasetName, CSVDatasetName

class QuantizationExperiment:
    """
    Orquestra um experimento completo de quantização.

    Segue o padrão de run_standard_experiment() existente:
    recebe (dataset_source, dataset_name) e trata do carregamento.

    Fluxo:
    1. Carregar dataset (via DatasetFactory)
    2. Treinar modelo baseline (float64)
    3. Para cada bit_width em config.data_bits:
       a. data_only  → FeatureQuantizer.fit(X_train) + transform + treinar + avaliar
       b. model_only → treinar normal + WeightQuantizer(model) + avaliar
       c. hybrid     → data_only + model_only combinados
    4. Compilar resultados como List[QuantizationResult]
    """

    def __init__(self, config: QuantizationConfig):
        self.config = config

    def run_supervised(
        self,
        dataset_source: DatasetSourceType,
        dataset_name: Union[SklearnDatasetName, CSVDatasetName],
        model_class: Type[BaseModel],
        model_params: dict,
        *,
        seed: int = 42,
    ) -> List[QuantizationResult]:
        """
        Executar para modelo supervisionado.

        Carrega dataset, faz train/test split com seed, e executa
        baseline + 3 modos × N bit-widths.
        """
        # ...

    def run_unsupervised(
        self,
        dataset_source: DatasetSourceType,
        dataset_name: Union[SklearnDatasetName, CSVDatasetName],
        model_class: Type[BaseModel],
        model_params: dict,
        *,
        seed: int = 42,
    ) -> List[QuantizationResult]:
        """
        Executar para modelo não-supervisionado.
        Métrica: silhouette_score (não requer labels_true).
        """
        # ...
```

#### 5.2 Case Study 4 (`scripts/case4.py`)

```python
"""
Case Study 4: SIP-Q - Impacto da Quantização em ML

Protocolo Experimental:
    12 combinações dataset×algoritmo × 3 bit-widths × 3 modos × 5 seeds
    = 540 execuções quantizadas
    + 60 execuções baseline (normal/float64, 1 por combinação×seed)
    = 600 medições totais

Datasets:       Wine, Digits (supervisionado), Make Blobs, Make Moons (clustering)
Algoritmos:     KNN, DT, MLP (supervisionado), MBK, GMM, AC (clustering)
Bit-widths:     8, 16, 32
Perfil padrão:  int (uint8/uint16/uint32)
Extensão:       float16 no ponto de 16 bits
Modos:          data_only, model_only, hybrid
Seeds:          42, 123, 456, 789, 1024
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Tuple, Type

sys.path.insert(0, str(Path(__file__).parent.parent))

from smart_inference_ai_fusion.core.base_model import BaseModel
from smart_inference_ai_fusion.models.knn_model import KNNModel
from smart_inference_ai_fusion.models.tree_model import DecisionTreeModel
from smart_inference_ai_fusion.models.mlp_model import MLPModel
from smart_inference_ai_fusion.models.minibatch_kmeans_model import MiniBatchKMeansModel
from smart_inference_ai_fusion.models.gaussian_mixture_model import GaussianMixtureModel
from smart_inference_ai_fusion.models.agglomerative_clustering_model import (
    AgglomerativeClusteringModel,
)
from smart_inference_ai_fusion.utils.types import (
    DatasetSourceType, SklearnDatasetName,
)
from smart_inference_ai_fusion.quantization.core import QuantizationConfig
from smart_inference_ai_fusion.experiments.quantization_experiment import QuantizationExperiment

# =============================================================================
# STUDY CONFIGURATION
# =============================================================================

SEEDS = [42, 123, 456, 789, 1024]

# Datasets — segue mesmo padrão (source, name, label) de case1/2/3
# NOTA: Make Blobs/Moons são datasets SKLEARN (gerados via sklearn.datasets)
SUPERVISED_DATASETS = [
    (DatasetSourceType.SKLEARN, SklearnDatasetName.WINE, "Wine"),
    (DatasetSourceType.SKLEARN, SklearnDatasetName.DIGITS, "Digits"),
]

UNSUPERVISED_DATASETS = [
    (DatasetSourceType.SKLEARN, SklearnDatasetName.MAKE_BLOBS, "MakeBlobs"),
    (DatasetSourceType.SKLEARN, SklearnDatasetName.MAKE_MOONS, "MakeMoons"),
]

# Algoritmos — usa classes do framework (não sklearn diretamente)
SUPERVISED_ALGOS: Dict[str, Tuple[Type[BaseModel], dict]] = {
    "KNN": (KNNModel, {"n_neighbors": 5}),
    "DT": (DecisionTreeModel, {"max_depth": 10, "random_state": None}),
    "MLP": (MLPModel, {"hidden_layer_sizes": (100,), "max_iter": 500,
                        "random_state": None}),
}

UNSUPERVISED_ALGOS: Dict[str, Tuple[Type[BaseModel], dict]] = {
    "MBK": (MiniBatchKMeansModel, {"n_clusters": 3, "random_state": None}),
    "GMM": (GaussianMixtureModel, {"n_components": 3, "random_state": None}),
    "AC": (AgglomerativeClusteringModel, {"n_clusters": 3}),
}


def main():
    parser = argparse.ArgumentParser(description="SIP-Q Case Study 4")
    parser.add_argument("--datasets", nargs="+",
                        default=["Wine", "Digits", "MakeBlobs", "MakeMoons"])
    parser.add_argument("--algorithms", nargs="+",
                        default=["KNN", "DT", "MLP", "MBK", "GMM", "AC"])
    parser.add_argument("--bits", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--method", default="uniform")
    parser.add_argument("--dtype-profile", default="integer",
                        choices=["integer", "float16"],
                        help="Perfil de tipo: 'integer' (core) ou 'float16'")
    parser.add_argument("--output", default="results/case4")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for seed in args.seeds:
        config = QuantizationConfig(
            data_bits=tuple(args.bits),
            model_bits=tuple(args.bits),
            dtype_profile=args.dtype_profile,
            method=args.method,
            enable_hybrid=True,
            random_seed=seed,
        )
        experiment = QuantizationExperiment(config)

        # --- Supervisionados ---
        for source, name, label in SUPERVISED_DATASETS:
            if label not in args.datasets:
                continue
            for algo_key in args.algorithms:
                if algo_key not in SUPERVISED_ALGOS:
                    continue
                model_class, kwargs = SUPERVISED_ALGOS[algo_key]
                algo_params = {**kwargs}
                if "random_state" in algo_params:
                    algo_params["random_state"] = seed
                results = experiment.run_supervised(
                    source, name,
                    model_class, algo_params,
                    seed=seed,
                )
                for r in results:
                    r.metadata.update({
                        "dataset": label, "algorithm": algo_key, "seed": seed,
                    })
                all_results.extend(results)

        # --- Não-supervisionados ---
        for source, name, label in UNSUPERVISED_DATASETS:
            if label not in args.datasets:
                continue
            for algo_key in args.algorithms:
                if algo_key not in UNSUPERVISED_ALGOS:
                    continue
                model_class, kwargs = UNSUPERVISED_ALGOS[algo_key]
                algo_params = {**kwargs}
                if "random_state" in algo_params:
                    algo_params["random_state"] = seed
                results = experiment.run_unsupervised(
                    source, name,
                    model_class, algo_params,
                    seed=seed,
                )
                for r in results:
                    r.metadata.update({
                        "dataset": label, "algorithm": algo_key, "seed": seed,
                    })
                all_results.extend(results)

    # Guardar resultados
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"case4_results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump([r.model_dump() for r in all_results], f, indent=2)

    print(f"\n✅ {len(all_results)} resultados guardados em: {output_file}")


if __name__ == "__main__":
    main()
```

Exemplo de uso:
```bash
# Teste rápido (1 dataset, 1 algoritmo, 1 seed — perfil integer)
python scripts/case4.py --datasets Wine --algorithms KNN --bits 16 --seeds 42

# Experimento completo core (540 execuções quantizadas + 60 baseline)
python scripts/case4.py \
    --datasets Wine Digits MakeBlobs MakeMoons \
    --algorithms KNN DT MLP MBK GMM AC \
    --bits 8 16 32 \
    --seeds 42 123 456 789 1024 \
    --output results/case4/

# Extensão float16 (apenas 16-bit, comparação direta com int16)
python scripts/case4.py \
    --datasets Wine Digits MakeBlobs MakeMoons \
    --algorithms KNN DT MLP MBK GMM AC \
    --bits 16 \
    --dtype-profile float16 \
    --seeds 42 123 456 789 1024 \
    --output results/case4_float16/
```

#### 5.3 Matriz Fatorial Completa e Controle Baseline

**Decisão de engenharia:** sim, quantizar o experimento inteiro e comparar sempre com execução normal.

Matriz principal (quantizada):

```
4 datasets × 6 algoritmos × 3 bit-widths (8/16/32) × 3 modos × 5 seeds
= 540 execuções quantizadas
```

Controle obrigatório (normal):

```
4 datasets × 6 algoritmos × 1 baseline float64 × 5 seeds
= 60 execuções baseline
```

Total de medições para análise estatística pareada:

```
540 + 60 = 600
```

Regras metodológicas:

- Baseline e quantizado devem usar **o mesmo split e seed**.
- A comparação principal é por `delta = score_quantizado - score_baseline`.
- Agregar por média, desvio padrão e intervalo de confiança por combinação.
- Reportar resultados separados para supervisionado (accuracy/F1) e clustering (silhouette/NMI).

#### 5.3.1 Como o `float16` será utilizado

Estratégia em duas etapas:

1. **Core (obrigatório):** executar apenas `int8/int16/int32` (600 medições totais).
2. **Extensão (opcional):** adicionar `float16` para comparar com `int16`.

Execução recomendada da extensão:

- Rodar `float16` primeiro em `data_only`.
- Se houver ganho relevante, expandir para `model_only` e `hybrid`.

Custo incremental:

- `float16` só em `data_only`: `+60` execuções quantizadas.
- `float16` em todos os 3 modos: `+180` execuções quantizadas.

Critério de decisão para manter `float16` no relatório final:

- Diferença média absoluta >= 1 ponto percentual em accuracy/silhouette contra `int16`.
- Ou melhoria estatisticamente significativa (teste pareado, p ajustado < 0.05).

#### 5.4 Integração com SIP + SIP-V

```python
# Experimento combinado: SIP + SIP-Q
# Pergunta: A degradação é aditiva ou multiplicativa?

baseline           = run_baseline(X, model)           # Sem nada
sip_only           = run_sip(X_perturbed, model)      # Só perturbação
quant_only         = run_quantized(X_quant, model)    # So quantização
combined           = run_sip_then_quant(X_perturbed_quant, model)

interaction = (baseline - combined) - (
    (baseline - sip_only) + (baseline - quant_only)
)
# Se interaction ~= 0  -> efeito aditivo
# Se interaction > 0   -> efeito multiplicativo (pior que a soma)
# Se interaction < 0   -> efeito sub-aditivo (modelos compensam)
```

#### Entregáveis:
- [ ] `QuantizationExperiment` para supervisionado e não-supervisionado
- [ ] `case4.py` funcional com CLI (argparse)
- [ ] Saída em JSON compatível com cases anteriores
- [ ] Testes de integração
- [ ] CI atualizado

---

### Fase 6 - Análise e Documentação (Dias 24-28)

**Objetivo:** Analisar os resultados experimentais, criar visualizações publicáveis
e documentar todo o trabalho realizado.

> **Dependência:** `matplotlib` e `seaborn` devem ser adicionados ao `pyproject.toml`
> como dependências opcionais (`[project.optional-dependencies] viz = ["matplotlib>=3.8", "seaborn>=0.13"]`)
> antes de iniciar esta fase.

#### 6.1 Análise de Resultados

**Metodologia de comparação (obrigatória):**

1. Comparação pareada por seed: baseline vs int8/int16/int32.
2. Quando habilitado, comparar também `int16` vs `float16` (mesmo seed e mesmo split).
3. Reportar `delta` absoluto e relativo (%).
4. Reportar intervalo de confiança (bootstrap 95%).
5. Aplicar teste estatístico pareado (Wilcoxon ou t-test pareado) por algoritmo.
6. Ajustar p-valor para múltiplas comparações (Holm-Bonferroni).

**Tabela principal esperada:**

| Dataset | Algoritmo | Método | 8-bit | 16-bit | 32-bit | Baseline |
|---------|-----------|--------|-------|--------|--------|----------|
| Wine | KNN | uniform | 0.85 | 0.93 | 0.94 | 0.94 |
| Wine | DT | uniform | 0.91 | 0.93 | 0.93 | 0.93 |
| Wine | MLP | uniform | 0.78 | 0.92 | 0.94 | 0.94 |
| MakeBlobs | MBK | uniform | 0.82* | 0.91* | 0.92* | 0.92* |
| MakeBlobs | GMM | uniform | 0.75* | 0.90* | 0.92* | 0.92* |

_* = silhouette score em vez de accuracy_

#### 6.2 Script de Análise (`scripts/analyze_case4_results.py`)

```python
"""Análise dos resultados do Case Study 4."""
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(results_dir: Path) -> pd.DataFrame:
    """Carregar todos os resultados JSON."""
    records = []
    for file in results_dir.glob("*.json"):
        with open(file) as f:
            data = json.load(f)
            for result in data:
                records.append(result)
    return pd.DataFrame(records)


def plot_accuracy_vs_bitwidth(df: pd.DataFrame):
    """Plotar acurácia vs largura de bits — uma linha por algoritmo."""
    plt.figure(figsize=(12, 6))
    for algo in df["algorithm_name"].unique():
        df_algo = df[df["algorithm_name"] == algo]
        plt.plot(
            df_algo["bit_width"],
            df_algo["quantized_accuracy"],
            marker="o", label=algo,
        )
    plt.xlabel("Largura de Bits")
    plt.ylabel("Acurácia")
    plt.title("Acurácia vs. Largura de Bits por Algoritmo")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/case4/accuracy_vs_bitwidth.png", dpi=300)


def plot_pareto_frontier(df: pd.DataFrame):
    """Plotar fronteira de Pareto (acurácia vs eficiência)."""
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        df["compression_ratio"],
        df["quantized_accuracy"],
        c=df["bit_width"], s=100, cmap="viridis", alpha=0.6,
    )
    plt.colorbar(scatter, label="Bits")
    plt.xlabel("Compressão (×)")
    plt.ylabel("Acurácia")
    plt.title("Fronteira de Pareto: Acurácia vs. Eficiência")
    plt.grid(True)
    plt.savefig("results/case4/pareto_frontier.png", dpi=300)


def plot_heatmap(df: pd.DataFrame):
    """Heatmap de degradação: Dataset × Algoritmo."""
    pivot = df.pivot_table(
        values="accuracy_degradation",
        index="dataset_name",
        columns="algorithm_name",
        aggfunc="mean",
    )
    plt.figure(figsize=(10, 6))
    import seaborn as sns
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn_r")
    plt.title("Degradação Média por Dataset × Algoritmo")
    plt.savefig("results/case4/heatmap_degradation.png", dpi=300)


if __name__ == "__main__":
    df = load_results(Path("results/case4"))
    plot_accuracy_vs_bitwidth(df)
    plot_pareto_frontier(df)
    plot_heatmap(df)
    print(f"✅ {len(df)} resultados analisados, gráficos gerados.")
```

#### 6.3 Visualizações Planeadas

1. **Curva Acurácia vs. Bits** — Uma linha por algoritmo, eixo X = bits, eixo Y = acurácia
2. **Fronteira de Pareto** — Scatter: compressão (X) vs. acurácia (Y), cor = algoritmo
3. **Heatmap** — Dataset × Algoritmo, cor = degradação média
4. **Barras agrupadas** — Comparação supervisionado vs. não-supervisionado

#### 6.4 Métricas de Robustez Adicionais

| Métrica | Fórmula | Descrição |
|---------|---------|-----------|
| **Bit sensitivity** | Slope de acurácia vs. bits | Quão sensível é o modelo a cada bit removido |
| **Prediction consistency** | Concordância baseline vs. quantizado | Percentagem de previsões idênticas |
| **Quantization error (MAE)** | mean(\|X - X_r\|) | Erro absoluto médio da reconstrução |

#### 6.5 Documentação

- [ ] `docs/SIP-Q_RESULTADOS.md` — Relatório técnico dos resultados
- [ ] README do módulo `quantization/`
- [ ] Tutorial Jupyter notebook
- [ ] Docstrings em todos os módulos públicos

#### 6.6 Melhorias de Arquitetura e Metodologia

1. **Manifesto de Reprodutibilidade**
    - Persistir versão de pacotes, hash de config e seeds em cada arquivo de resultado.
2. **Execução Resumível (checkpoint)**
    - Identificador único por execução (`dataset+algoritmo+modo+bits+seed`) para retomar runs interrompidos.
3. **Validação anti-vazamento de dados**
    - Ajustar parâmetros de quantização apenas no conjunto de treino em cenários supervisionados.
4. **Camada de métricas unificada**
    - Interface única para `accuracy`, `F1`, `silhouette`, `NMI` e tempo/memória.
5. **Governança de matriz de compatibilidade**
    - Tabela explícita de pares válidos dataset×algoritmo para evitar execução inválida.

#### Entregáveis:
- [ ] 4+ visualizações (accuracy vs. bits, Pareto, heatmap, barras)
- [ ] Relatório técnico com tabelas e gráficos
- [ ] Documentação atualizada
- [ ] Pylint >= 9.5/10 em todos os módulos

---

## 🗂️ Seleção de Datasets

### Recomendação: 4 Datasets (2 + 2)

Após análise do framework, recomendo **4 datasets** - a quantidade ideal para:
- Cobrir supervised + unsupervised sem ser excessivo
- Reutilizar datasets já integrados (menor custo de implementação)
- Gerar resultados suficientes para conclusões estatísticas
- Manter o tempo de execução razoável (540 runs com 5 seeds)

> **Por que não mais?** Com 6 algoritmos x 3 bit-widths x 3 modos x 5 seeds, cada
> dataset adicional gera 270 execuções. 4 datasets × 6 algoritmos = 12 combinações × 3 × 3 × 5 = 540 runs. Com 6 datasets seria bem mais -
> excessivo para uma primeira extensão.

### Datasets Selecionados

#### Supervisionados (Classificação)

| Dataset | Amostras | Features | Classes | Porque |
|---------|----------|----------|---------|--------|
| **Wine** | 178 | 13 | 3 | Features contínuas com ranges variados (0.7 a 1680), já usado em case1-3. Excelente para testar sensibilidade da quantização a escalas diferentes. |
| **Digits** | 1797 | 64 | 10 | Features de imagem (0-16), alta dimensionalidade. O domínio mais relevante para quantização na prática (imagens são naturalmente quantizadas). Já usado em case1-2. |

#### Não-Supervisionados (Clustering)

| Dataset | Amostras | Features | Clusters | Porque |
|---------|----------|----------|----------|--------|
| **Make Blobs** | 500 | 2 | 3 | Clusters isotrópicos (esféricos). Baseline controlado - se a quantização falha aqui, há um problema. Já usado em case3. |
| **Make Moons** | 500 | 2 | 2 | Clusters não-lineares (forma de lua). Testá se a quantização destroi a estrutura nao-linear. Já usado em case2. |

### Justificacao da Seleção

```
                    Dimensionalidade
                    Baixa (2D)      Média (13D)     Alta (64D)
                +---------------+---------------+---------------+
  Supervisado   |               |   Wine [x]    |   Digits [x]  |
                +---------------+---------------+---------------+
  Nao-Superv.   | Make Blobs [x]|               |               |
                | Make Moons [x]|               |               |
                +---------------+---------------+---------------+
```

**Cobertura:**
- [x] 2 tipos de tarefa (classificação + clustering)
- [x] 3 níveis de dimensionalidade (2D, 13D, 64D)
- [x] 100% já integrados no framework (0 dependências novas)
- [x] Range de complexidade: blobs simples -> digits com 10 classes
- [x] Range de tamanho: 178 -> 1797 amostras

---

## 🤖 Seleção de Algoritmos

### Recomendação: 6 Algoritmos (3 Supervisionados + 3 Não-Supervisionados)

A divisão **3 + 3** é a melhor escolha porque:

1. **Equilíbrio** - Mesmo peso para supervised e unsupervised na análise
2. **Diversidade** - Cobre parametric, non-parametric e probabilistic
3. **Praticidade** - Todos já existem no framework (modelos já implementados)
4. **Impacto variado** - Desde "quase imune" (DT) ate "muito sensível" (GMM)

> **Por que não 2+2+2 (classif + regr + cluster)?** O framework não tem infraestrutura
> forte de regressão (so Ridge e RandomForestRegressor). Os case studies existentes focam
> em classificação e clustering. Manter 3+3 e mais coerente com o trabalho já feito.

### Algoritmos Selecionados

#### Supervisionados (3)

| # | Algoritmo | Tipo | Classe no Framework | Sensibilidade a Quantização | Porque |
|---|-----------|------|---------------------|---------------------------|--------|
| 1 | **KNN** | Non-parametric | `KNNModel` | **Alta** | Baseia-se em distâncias euclidianas - pequenas alterações nos valores mudam quais são os vizinhos mais próximos. Armazena todos os dados de treino (`_fit_X`), logo a compressão de memória e maxima. |
| 2 | **Decision Tree** | Tree-based | `DecisionTreeModel` | **Baixa** | Usa limiares discretos (`threshold`). A quantização so afeta se o arredondamento mover o valor para o lado errado do limiar. E o "controlo negativo" - espera-se pouca degradação. |
| 3 | **MLP** | Neural Network | `MLPModel` | **Alta** | Tem o maior número de parâmetros (`coefs_` e `intercepts_` por camada). Muito estudado em deep learning quantization - serve de ponte com a literatura existente. |

#### Não-Supervisionados (3)

| # | Algoritmo | Tipo | Classe no Framework | Sensibilidade a Quantização | Porque |
|---|-----------|------|---------------------|---------------------------|--------|
| 4 | **MiniBatchKMeans** | Centroid-based | `MiniBatchKMeansModel` | **Média** | Centróides (`cluster_centers_`) são médias dos pontos - alguma tolerância ao ruido. Mas a atribuição de clusters depende de distâncias, tal como KNN. |
| 5 | **GMM** | Probabilistic | `GaussianMixtureModel` | **Alta** | Opera com médias **e covariâncias** (`means_`, `covariances_`). As covariâncias envolvem valores muito pequenos e diferenças subtis - muito sensível a perda de precisão. |
| 6 | **Agglomerative** | Hierarchical | `AgglomerativeClusteringModel` | **Média** | Não tem pesos internos (recalcula a árvore de distâncias). E afetado apenas pela quantização dos dados de entrada. Complementa MBK e GMM como abordagem fundamentalmente diferente. |

### Impacto Esperado

```
Sensibilidade a Quantização (8-bit)

Alta      |  MLP ====================  (muitos pesos)
          |  GMM ==================    (covariâncias sensiveis)
          |  KNN ================      (distâncias afetadas)
Média     |  MBK ==========           (centróides tolerantes)
          |  AC  ========             (recalcula distâncias)
Baixa     |  DT  ====                (limiares discretos)
          +------------------------------------
```

### Matriz Completa: Datasets x Algoritmos

| | Wine (13D, 3c) | Digits (64D, 10c) | Make Blobs (2D, 3c) | Make Moons (2D, 2c) |
|---|---|---|---|---|
| **KNN** | [x] | [x] | - | - |
| **DT** | [x] | [x] | - | - |
| **MLP** | [x] | [x] | - | - |
| **MBK** | - | - | [x] | [x] |
| **GMM** | - | - | [x] | [x] |
| **AC** | - | - | [x] | [x] |

**Total quantizado: 12 combinações x 3 bit-widths x 3 modos x 5 seeds = 540 execuções**

**Total baseline (normal/float64): 12 combinações x 1 x 1 x 5 seeds = 60 execuções**

**Total geral para análise: 600 medições**

---

## 🔬 Questões de Pesquisa e Hipóteses

### Questões de Pesquisa

| ID | Questão |
|----|---------|
| **RQ1** | Qual é o impacto da quantização de dados (uint8/uint16/uint32) na accuracy de modelos supervisionados e na silhouette de modelos de clustering? |
| **RQ2** | Existem diferenças significativas de robustez à quantização entre algoritmos supervisionados e não-supervisionados? |
| **RQ3** | Qual é o bit-width mínimo que preserva a qualidade preditiva dentro de uma margem aceitável (≤ 5% de degradação)? |
| **RQ4** | O efeito combinado de perturbações SIP e quantização SIP-Q é aditivo ou existe interação entre os dois fatores? |

### Hipóteses Estatísticas

| RQ | H₀ (Nula) | H₁ (Alternativa) | Teste |
|----|-----------|-------------------|-------|
| **RQ1** | Não há diferença significativa entre a métrica baseline (float64) e a métrica após quantização a 16 bits | A quantização a 16 bits degrada significativamente a métrica | Wilcoxon signed-rank (pareado, 5 seeds) |
| **RQ2** | A degradação média dos algoritmos supervisionados é igual à dos não-supervisionados | Existe diferença significativa entre os dois grupos | Mann-Whitney U |
| **RQ3** | Todos os bit-widths (8, 16, 32) produzem degradação equivalente | Existe pelo menos um bit-width com degradação significativamente diferente | Friedman + post-hoc Nemenyi |
| **RQ4** | O efeito de SIP e SIP-Q combinados é a soma dos efeitos individuais (aditivo) | Existe interação entre SIP e SIP-Q (efeito não-aditivo) | ANOVA two-way (Fase 5) |

> **Nota:** Testes não-paramétricos (Wilcoxon, Mann-Whitney, Friedman) são preferidos
> dado o número reduzido de repetições (5 seeds) e a ausência de garantia de normalidade.

---

## ✅ Critérios de Sucesso

### Técnicos

| Critério | Alvo | Como Medir |
|----------|------|------------|
| Degradação 16-bit | < 5% accuracy / < 0.05 silhouette | Média dos 5 seeds |
| Redução de memória 8-bit | >= 75% (float64->int8 = 8x) | `X.nbytes` antes/depois |
| Overhead quantização 8-bit | < 15% do tempo baseline | Mediana de 50 runs (`t_quant + t_infer` vs `t_baseline`) |
| Cobertura de testes | >= 95% | pytest --cov |
| Qualidade de código | Pylint >= 9.5/10 | `pylint quantization/` |
| Reprodutibilidade | 5 seeds | Seeds: 42, 123, 456, 789, 1024 |

### Investigação

| RQ | Questão | Validação |
|----|---------|-----------|
| **RQ1** | Quais algoritmos são mais robustos? | Ranking por degradação média + Wilcoxon pareado |
| **RQ2** | Supervisionados vs. não-supervisionados? | Mann-Whitney U entre os dois grupos |
| **RQ3** | Qual o bit-width ótimo? | Friedman + post-hoc Nemenyi; ponto de inflexão na curva |
| **RQ4** | Efeito SIP+SIP-Q é aditivo? | ANOVA two-way (Fase 5) |

### Integração

| Critério | Alvo |
|----------|------|
| CI/CD | Todos os testes passando no GitHub Actions |
| Compatibilidade | Funciona com SIP e SIP-V existentes |
| Documentação | API docs + tutorial + relatório |

---

## ⚠️ Ameaças à Validade

### Validade Interna

| Ameaça | Mitigação |
|--------|-----------|
| **Vazamento de dados na calibração** | `FeatureQuantizer.fit()` calibrado apenas no treino; `transform()` aplicado separadamente ao teste. Validação anti-vazamento incluída nos testes da Fase 3. |
| **Parâmetros de quantização dependentes do seed** | Seed controla apenas train/test split e inicialização de modelos estocásticos (MLP, GMM, MBK). Parâmetros de quantização derivam deterministicamente dos dados. |
| **Overfitting do método ao dataset** | Três métodos de quantização independentes (uniform, kmeans, percentile) reduzem risco de conclusão espúria. |

### Validade Externa

| Ameaça | Mitigação |
|--------|-----------|
| **Poucos datasets (4)** | Representam 4 cenários distintos: alta dimensionalidade (Digits, 64D), média dimensionalidade (Wine, 13D), dados sintéticos bem-separados (Blobs), e dados com fronteira não-linear (Moons). |
| **Apenas algoritmos clássicos** | Escopo deliberado: o framework SIP foca em ML clássico. Deep learning é out-of-scope. |
| **Ausência de dados tabulares reais de grande escala** | Limitação reconhecida. Trabalho futuro pode incluir datasets UCI maiores. |

### Validade de Construto

| Ameaça | Mitigação |
|--------|-----------|
| **Roundtrip não simula hardware real** | Explicitamente documentado como simulação de erro de quantização. O objetivo é medir robustez algorítmica, não performance de hardware. |
| **sklearn converte tudo para float64** | Medição de overhead temporal inclui ciclo completo (quantize + dequantize + inferência). Reportado como `overhead_pct`, não como speedup. |

### Validade de Conclusão

| Ameaça | Mitigação |
|--------|-----------|
| **Múltiplas comparações** | Correção de Holm-Bonferroni aplicada quando se comparam mais de 2 condições (RQ3). |
| **Tamanho amostral reduzido (5 seeds)** | Testes não-paramétricos (Wilcoxon, Mann-Whitney, Friedman) são usados. Effect size (Cohen's d) reportado em todas as comparações. |

---

## 📐 Decisões Metodológicas

### Normalização vs. Quantização Direta

Quando as features têm escalas muito diferentes (e.g., Wine: Alcohol ∈ [11,15] vs. Proline ∈ [278,1680]),
a quantização uniform distribui os níveis de forma desigual entre features.

**Decisão adoptada:** Quantizar os dados **sem normalização prévia**, para medir o impacto real
da quantização nos dados tal como existem. A normalização (StandardScaler / MinMaxScaler) é uma
transformação independente que o pipeline de ML pode aplicar antes ou depois — misturá-la com
a quantização confundiria os dois efeitos.

**Justificação:** O objetivo de SIP-Q é medir a robustez dos algoritmos ao erro de quantização,
não otimizar a accuracy. Se normalizássemos antes de quantizar, estaríamos a medir a robustez
dos dados normalizados+quantizados — um efeito confundido.

**Exceção:** O método `minmax_quantize` já inclui normalização implícita (mapeia para [0,1]
antes de quantizar). Isto é reportado nos resultados e discutido como variante.

> **Nota para a dissertação:** Esta decisão deve ser justificada na secção de Metodologia,
> referenciando a prática padrão em benchmarks de quantização (e.g., TensorFlow Lite
> quantiza pesos e ativações no espaço original, não normalizado).

---

## 🎯 Próximos Passos

### Pré-requisitos (Antes de Iniciar)
1. Revisar e aprovar este plano (versão final)
2. Criar branch `feature/sip-q-quantization`
3. Verificar que CI/CD está verde na branch `main`

### Bloco 1 — Fundação (Fases 0-1)
1. Setup do módulo `quantization/` com estrutura de diretórios e `__init__.py`
2. Implementar `QuantizationConfig`, `QuantizationResult`, tipos (`DTypeProfile`, `BitWidth`, `QuantMethod`)
3. Primeiros testes unitários (>= 10 testes para classes base)

### Bloco 2 — Quantização Core (Fases 2-3)
1. Implementar 4 métodos de quantização de features (uniform, min-max, k-means, percentile)
2. Implementar `FeatureQuantizer` com suporte a `dtype_profile` ("integer" + "float16")
3. Implementar `WeightQuantizer` para os 6 algoritmos
4. Primeiros benchmarks com Wine (dataset mais simples)

### Bloco 3 — Experimento e Integração (Fases 4-5)
1. Métricas supervisionadas e clustering
2. `case4.py` funcional com CLI (incluindo `--dtype-profile`)
3. Executar 600 medições core (12 combos × 3 bits × 3 modos × 5 seeds + baseline)
4. Executar extensão float16 (60-180 medições adicionais)
5. Integração com SIP + SIP-V: teste de aditividade

### Bloco 4 — Análise e Publicação (Fase 6)
1. Visualizações: curvas acurácia vs. bits, Pareto, heatmaps
2. Análise estatística: teste pareado, Holm-Bonferroni, bootstrap 95%
3. Comparação int16 vs. float16 (se extensão executada)
4. Relatório técnico (`docs/SIP-Q_RESULTADOS.md`)

---

## 📚 Referências

### Artigos Académicos

1. **Jacob et al. (2018)** - "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"
2. **Gong et al. (2014)** - "Compressing Deep Convolutional Networks using Vector Quantization"
3. **Han et al. (2016)** - "Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding"

### Ferramentas

- PyTorch Quantization (`torch.quantization`)
- TensorFlow Lite (Model Optimization Toolkit)
- ONNX Runtime Quantization

---

**Versão:** 4.1 (Versão Final Revista)
**Data:** 4 de Março de 2026
**Status:** Aprovado para Implementação
