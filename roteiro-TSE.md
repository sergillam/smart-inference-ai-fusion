# 📄 Documento Mestre do Projeto: Plataforma SIP para IEEE TSE (Qualis A1)

## 📌 Contexto Global (System Prompt para IAs Assistentes)
**Aja como um Engenheiro de Software Sênior e Pesquisador de Ciência da Computação Empírica.**
O objetivo deste projeto é refatorar e instrumentar a plataforma "SIP" (Smart Inference Pipeline) e os seus submódulos "SIP-V" (Verificação Formal com SMT) e "SIP-Q" (Quantização). O código gerado deve atingir o rigor técnico e científico exigido pela revista *IEEE Transactions on Software Engineering (TSE)*.
O foco da implementação é: tolerância a falhas (*crash-safe*), isolamento de recursos via contêineres Ubuntu, coleta de métricas exaustivas (tempo, RAM, constraints SMT, colapso estrutural) e validação estatística rigorosa baseada em múltiplas execuções estocásticas (configurável, padrão de 30 *seeds* aleatórias).

---

# PARTE I: PROTOCOLO EXPERIMENTAL (A Ciência)
*Esta seção define as regras, parâmetros e a metodologia do estudo empírico que irão para a seção "Study Design" do artigo.*

## 1. Obtenção e Preparação das Bases de Dados
As bases foram escolhidas pelo seu caráter de missão crítica e alta dimensionalidade. Para garantir a rastreabilidade (exigência IEEE), ambas devem ser versionadas utilizando o **DVC (Data Version Control)**.

* **WiDS Datathon 2020 / ICU Mortality Prediction (Saúde / UTI):**
  * *Domínio:* Previsão de risco/mortalidade baseada em sinais vitais e exames laboratoriais.
  * *Aquisição:* Kaggle API (`kaggle competitions download -c widsdatathon2020`).
  * *Pré-processamento:* Imputação de dados em falta e normalização *Min-Max* (escalonamento estrito entre [0, 1] é crítico para limitar o espaço de busca dos solvers SMT).
* **IEEE-CIS Fraud Detection (Finanças):**
  * *Domínio:* Detecção de transações fraudulentas em e-commerce (altamente desbalanceado).
  * *Aquisição:* Kaggle API (`kaggle competitions download -c ieee-fraud-detection`).
  * *Pré-processamento:* *Feature Selection* estrita (reter apenas as 30 a 50 variáveis mais importantes para evitar explosão de estados no Z3), *One-Hot Encoding* para categorias pequenas, *Hashing* para grandes e preenchimento de nulos.sim

## 2. Algoritmos e Parametrização Fixa
A hiperparametrização será **estática**. A variação provirá estritamente das *seeds* de inicialização para evitar variáveis de confusão.
1. **Regressão Logística (LR):** `penalty='l2'`, `C=1.0`, `solver='lbfgs'`, `max_iter=1000`. (*Baseline linear*).
2. **Árvore de Decisão (DT):** `criterion='gini'`, **`max_depth=5`**. (*Modelo simbólico. Limite de profundidade vital para tratabilidade no Z3*).
3. **Random Forest (RF):** `n_estimators=50`, `max_depth=5`, `bootstrap=True`. (*Causador proposital da explosão de estados no SIP-V*).
4. **Multi-Layer Perceptron (MLP):** `hidden_layer_sizes=(64, 32)`, `activation='relu'`, `solver='adam'`, `max_iter=500`. (*Teste de resiliência de pesos contínuos*).

## 3. Protocolo de Execução e Isolamento
* **Ambiente de Isolamento:** Contêiner baseado em `ubuntu:24.04` (Python 3.12). Limite estrito a `cpus: 2.0` e `mem_limit: 8G` via Docker Compose para evitar ruídos do SO no *wall_clock_time*.
* **Sementes e Repetições:** Execução configurável via variável `NUM_SEEDS` (Padrão = 30 repetições independentes). O código deve forçar `random_state=seed` na divisão de dados (`StratifiedShuffleSplit`), inicialização de modelos e injeção de ruído.
* **Matriz de Execução:** Para cada combinação (Dataset $\times$ Modelo $\times$ Seed), o fluxo rodará: 1) Baseline, 2) SIP Puro, 3) SIP-V (Z3/CVC5), 4) SIP-Q (int8), 5) Indústria (*Great Expectations*).

## 4. Validação Estatística (Análise Analítica)
O script final de análise aplicará:
1. **Teste de Normalidade:** Shapiro-Wilk.
2. **Teste de Significância:** Wilcoxon Signed-Rank ($\alpha = 0.05$) para tempos emparelhados.
3. **Tamanho do Efeito:** Vargha-Delaney ($\hat{A}_{12}$) para medir a relevância prática da diferença de overhead.
4. **Exportação:** Geração de tabelas direto para formato LaTeX.

---

# PARTE II: ROTEIRO DE ENGENHARIA (A Implementação)
*Prompts prontos para alimentar o GitHub Copilot/Cursor e construir a arquitetura descrita na Parte I.*

## 🛠️ Passo 1: Infraestrutura Relacional de Telemetria (Crash-Safe)
**Prompt para a IA (Foco no `telemetry_db.py`):**
> "Crie um módulo `telemetry_db.py` usando `SQLAlchemy`. O banco de dados deve ser SQLite (`experiments_tse.db`) configurado com `PRAGMA synchronous=NORMAL` e `journal_mode=WAL` para suportar alta concorrência.
> 1. Defina as seguintes tabelas:
>    - `sipv_metrics`: (execution_id, dataset, model, seed, wall_clock_ms, cpu_time_ms, translation_time_ms, peak_ram_mb, num_constraints, num_vars, status, constraint_violated_type).
>    - `sipq_metrics`: (execution_id, dataset, model, seed, f1_score_float32, f1_score_int8, roc_auc_int8, decision_flip_rate, mse_reconstruction, wasserstein_distance, model_size_original, model_size_quantized).
>    - `fault_injection_metrics`: (execution_id, dataset, model, seed, injected_fault_type, caught_by_sipv, caught_by_great_expectations).
> 2. Implemente a função `log_metrics(table_name, data_dict)` garantindo um `commit` atômico a cada execução para evitar corrupção em caso de *Out Of Memory (OOM)*."

## 🛠️ Passo 2: Instrumentação Profunda do SIP-V
**Prompt para a IA (Foco na integração SMT):**
> "Refatore a função de chamada aos solvers SMT (Z3/CVC5). Crie um decorador `@profile_smt_execution` que extraia:
> 1. O tempo de tradução da árvore/modelo para fórmula lógica (`translation_time_ms`).
> 2. O tempo de resolução do solver (`wall_clock_ms` e `cpu_time_ms` via `os.times()`).
> 3. O pico de consumo de memória do processo específico do solver (use `psutil` mapeando o PID).
> 4. Número de `constraints` e `vars` interceptando o contexto SMT.
> 5. Trate os `Timeouts` de forma elegante (não quebre a execução) e salve o status e métricas na tabela `sipv_metrics` através do `telemetry_db.py`."

## 🛠️ Passo 3: Instrumentação do SIP-Q
**Prompt para a IA (Foco na Quantização):**
> "Adicione telemetria avançada à etapa de quantização para `int8`. Calcule e salve na tabela `sipq_metrics`:
> 1. Degradação de `f1_score` e `roc_auc`.
> 2. `decision_flip_rate`: Porcentagem de predições que mudaram puramente pelo truncamento numérico (comparação index a index).
> 3. `wasserstein_distance` (usando `scipy.stats.wasserstein_distance`) comparando a distribuição de probabilidades original vs quantizada.
> 4. Tamanho dos artefatos em bytes antes e depois."

## 🛠️ Passo 4: Implementação do Baseline da Indústria
**Prompt para a IA (Foco no `baseline_runner.py`):**
> "Crie o arquivo `baseline_runner.py` para ser o comparativo contra o SIP-V.
> 1. Configure a biblioteca `great_expectations` programaticamente.
> 2. Crie uma *Expectation Suite* que espelhe as regras de `bounds` e `type_safety` que utilizamos no Z3.
> 3. Rode a validação contra os dados que sofreram perturbação estocástica.
> 4. Capture o `wall_clock_time` e verifique se o erro foi pego (`caught_by_great_expectations`). Exporte para `fault_injection_metrics`."

## 🛠️ Passo 5: DevOps e Reprodutibilidade Aberta
**Prompt para a IA (Foco em Docker e Bash):**
> "Crie a infraestrutura de reprodutibilidade para selos IEEE.
> 1. Crie um `Dockerfile` leve (`ubuntu:24.04`, Python 3.12) com dependências nativas para compilar Z3/CVC5.
> 2. Crie um `docker-compose.yml` limitando recursos (`cpus: 2.0`, `mem_limit: 8G`).
> 3. Crie o script `run_tse_experiments.sh`. Ele deve ler a variável de ambiente `NUM_SEEDS` (com default 30). Faça três loops aninhados: Datasets (`wids-datathon-2020`, `ieee-cis`), Modelos (`lr`, `dt`, `rf`, `mlp`) e Sementes (de 1 a `$NUM_SEEDS`).
> 4. Implemente um *Circuit Breaker* no shell: capture *exit codes* como `137` (OOM Killer). Registre em um `error.log` e force a continuidade (`continue`) do loop."

## 🛠️ Passo 6: Análises Estatísticas e Geração de LaTeX
**Prompt para a IA (Foco no `statistical_analysis.py`):**
> "Crie o script `statistical_analysis.py` para testes de hipóteses de Engenharia de Software.
> 1. Leia o banco de dados `experiments_tse.db` via `pandas`.
> 2. Aplique o Teste de Wilcoxon Signed-Rank (`scipy.stats.wilcoxon`) comparando os tempos do SIP-V vs Great Expectations, e entre os modelos.
> 3. Calcule o Effect Size utilizando a métrica de Vargha e Delaney (A12).
> 4. A saída final não deve ser apenas print, mas gerar um arquivo `tabela_resultados.tex` com o código LaTeX formatado pronto para ser inserido no artigo da TSE."
