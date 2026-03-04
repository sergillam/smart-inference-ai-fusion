# 🚀 CI/CD Pipeline - GitHub Actions

Este arquivo documenta o pipeline de integração contínua (CI/CD) configurado para o projeto Smart Inference AI Fusion.

## 📋 Overview

O pipeline CI é acionado automaticamente em:
- ✅ **Push** para qualquer branch (`main`, `develop`, `feature/*`, `refactor/*`, `fix/*`)
- ✅ **Pull Requests** para as branches `main` ou `develop`

## 🔧 Jobs do Pipeline

### 1. **Tests & Quality Checks** (`tests`)
```
Executa:
├── 📦 Setup Python (3.10, 3.11, 3.12 testing)
├── 🔍 Pylint - Verificação de qualidade de código
│  └── Alvo: z3_plugin.py, cvc5_plugin.py, data_utils.py
│  └── Threshold: ≥ 9.5/10
├── 🧪 Pytest - Execução de todos os testes
│  └── 31 testes executados
│  └── Requer 100% de sucesso
└── 📊 Coverage Report
   └── Gera relatório de cobertura de código

Status required: ✅ DEVE PASSAR
```

### 2. **Code Quality Check** (`code-quality`)
```
Executa:
├── 📝 Verificação de .pylintrc
├── 🎯 Core Modules Quality Block
│  ├── Verifica z3_plugin.py
│  ├── Verifica cvc5_plugin.py
│  └── Verifica data_utils.py
└── 📊 Relatório detalhado

Status: ⚠️ Exit-zero (não bloqueia)
```

### 3. **Integration Tests** (`integration`)
```
Executa: (Depende de tests passar)
├── Case 1: SIP Architecture (Iris + KNN)
├── Case 2: Paradigm Comparison (Wine + KNN)
└── Case 3: Formal Verification (Wine + LR)

Status: ⚠️ Continue on error (testes opcionais)
```

### 4. **Notification** (`notify`)
```
Resumo final do pipeline com status de todos os jobs
```

---

## 📊 Fluxo de Execução

```
GitHub Event (push/PR)
         ↓
┌────────────────────────────────────┐
│  Tests & Quality Checks (parallel) │  ← CRÍTICO
│  - Pylint ≥ 9.5/10                │
│  - Pytest 31/31 passing            │
│  - Python 3.10, 3.11, 3.12         │
└────────────────────────────────────┘
         ↓ (se passar)
┌────────────────────────────────────┐
│     Code Quality Check (extra)      │  ← INFO
│  - Relatório detalhado Pylint     │
└────────────────────────────────────┘
         ↓ (paralelo)
┌────────────────────────────────────┐
│    Integration Tests (opcional)     │  ← EXTRA
│  - Case Studies 1, 2, 3             │
└────────────────────────────────────┘
         ↓
┌────────────────────────────────────┐
│        Notification Summary         │  ← REPORT
│  - Status geral do pipeline         │
└────────────────────────────────────┘
```

---

## ✅ Critérios de Sucesso

Para uma branch ser mergeada para `main`, o pipeline DEVE:

1. **✅ Pylint**: Score ≥ 9.5/10 nos módulos core
2. **✅ Pytest**: Todos os 31 testes passando
3. **✅ Python**: Compatibilidade com 3.10, 3.11, 3.12
4. ⚠️ **Coverage**: Gerar relatório (informativo)
5. ⚠️ **Integration**: Case studies rodarem (opcional)

---

## 🚀 Como Usar

### Enviar Branch com Push

```bash
# Cria branch local
git checkout -b feature/novo-recurso

# Faz alterações e commita
git add .
git commit -m "feat: novo recurso"

# Push da branch
git push origin feature/novo-recurso
```

**O que acontece:**
- GitHub Actions dispara automaticamente
- Executa todos os jobs do pipeline
- Exibe resultado no GitHub ✅ ou ❌

### Abrir Pull Request para main

```bash
# No GitHub, cria PR da sua branch para main
```

**O que acontece:**
- GitHub Actions executa pipeline novamente
- PR mostra status do pipeline verificado ✓
- Requer que testes passem antes de mergear (se configurado)

### Ver Status do Pipeline

1. Vá para **GitHub → Actions**
2. Selecione o workflow **"CI - Tests & Code Quality"**
3. Clique na execução (commit/PR)
4. Veja os detalhes de cada job

### Logs e Detalhes

Para cada job, você pode:
- Ver logs em tempo real ⏱️
- Fazer download de artefatos (coverage, resultados)
- Analisar failures específicas

---

## 📝 Configuração Disponível

### Branches que Acionam CI

```yaml
on:
  push:
    branches: [ main, develop, 'feature/**', 'refactor/**', 'fix/**' ]
  pull_request:
    branches: [ main, develop ]
```

**Adicione sua branch ao padrão se necessário.**

### Versões Python Testadas

```yaml
strategy:
  matrix:
    python-version: ['3.10', '3.11', '3.12']
```

**Altere se quiser testar outras versões.**

### Threshold de Qualidade

```yaml
pylint --fail-under=9.5
```

**Aumentar/diminuir conforme necessário.**

---

## 🔍 Monitoramento

### GitHub Actions Dashboard

- **URL**: `https://github.com/<owner>/<repo>/actions`
- **Status Badge**: Pode ser adicionado ao README

### Badges no README

```markdown
![CI Status](https://github.com/<owner>/<repo>/workflows/CI%20-%20Tests%20%26%20Code%20Quality/badge.svg)
```

---

## 🚨 Troubleshooting

### Teste Local Passa mas CI Falha

1. **Verificar Python version:**
   ```bash
   python --version
   # Deve ser 3.10+
   ```

2. **Verificar requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Rodar Pylint localmente:**
   ```bash
   pylint smart_inference_ai_fusion/verification/plugins/z3_plugin.py \
           smart_inference_ai_fusion/verification/plugins/cvc5_plugin.py \
           smart_inference_ai_fusion/verification/utils/data_utils.py
   ```

4. **Rodar pytest localmente:**
   ```bash
   pytest tests/ -v
   ```

### CI Muito Lento

- A estratégia matrix testa 3 versões Python (paralelo)
- Tempo esperado: ~10-15 minutos
- Pode ser otimizado removendo versões se necessário

### Ainda Falhando?

1. Clique no job falho no GitHub Actions
2. Veja os logs detalhados
3. Reproduza localmente
4. Fixe o issue
5. Push da correção dispara CI novamente

---

## 📈 Métricas Coletadas

O pipeline coleta:
- ✅ Pylint scores (9.5+/10)
- ✅ Pytest results (31 testes)
- ✅ Code coverage (%) 
- ✅ Case study results (JSON)
- ✅ Build time (minutos)

---

## 🎯 Próximos Passos (Opcional)

Para melhorar ainda mais o pipeline:

1. **Adicionar CodeCov Integration**: Badge de cobertura no README
2. **Adicionar Dependabot**: Verificação automática de dependências
3. **Adicionar SAST**: Segurança estática (CodeQL)
4. **Auto-deploy**: Deploy automático se testes passarem
5. **Slack Notifications**: Noticações em tempo real

---

## 📞 Suporte

Se o pipeline tiver problemas:
1. Verifique os logs no GitHub Actions
2. Reproduza localmente
3. Abra uma issue descrevendo o problema
4. Mencione o commit/workflow que falhou

---

**Última atualização:** 4 de março de 2026  
**Workflow file:** `.github/workflows/ci.yml`
