# Tech Challenge — Fase 1
## Sistema de Suporte ao Diagnóstico: Detecção de Câncer de Mama com Machine Learning

**FIAP — Pós Tech (IADT)**

Projeto acadêmico da Fase 1 do Tech Challenge: um pipeline de Machine Learning para apoiar o diagnóstico de câncer de mama (maligno vs. benigno) a partir de características extraídas de imagens digitalizadas de biópsias.

---

## 📁 Sobre o dataset

- **Fonte:** [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data) (Kaggle/UCI)
- **Arquivo:** `data.csv`, incluído neste repositório — não é necessário download manual nem credenciais
- 569 amostras, 30 features numéricas (raio, textura, perímetro, área, suavidade, concavidade etc.), sem valores ausentes
- Variável alvo: `diagnosis` (0 = maligno, 1 = benigno)

## 🗂️ Estrutura do repositório

```
├── tech_challenge_fase1.ipynb   # notebook principal
├── data.csv                     # dataset
└── README.md
```

---

## ▶️ Instruções de Execução

Existem duas formas de rodar este projeto: **Google Colab** (mais simples, sem instalar nada localmente) ou **ambiente local** (Jupyter/VS Code).

### Opção 1 — Google Colab

1. Acesse [colab.research.google.com](https://colab.research.google.com)
2. Faça upload do notebook `tech_challenge_fase1.ipynb` (**Arquivo → Fazer upload de notebook**)
3. Execute as células em ordem (**Ambiente de execução → Executar tudo**)
4. Na célula de carga dos dados:
   - Se o repositório inteiro (incluindo `data.csv`) não tiver sido carregado junto, o notebook detecta que o arquivo não está presente e abre automaticamente um widget de upload — basta selecionar o `data.csv` local
5. A primeira célula já cuida da instalação de dependências extras (`!pip install -q shap`) — não é necessário nenhum passo manual

### Opção 2 — Ambiente local (Jupyter / VS Code)

**Pré-requisitos:** Python 3.10+ instalado

**Passo 1 — Clone o repositório**
```bash
git clone https://github.com/romulooliverr/Tech-Challenge-GP15
cd Tech-Challenge-GP15
```

**Passo 2 — Crie um ambiente virtual (recomendado)**
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

**Passo 3 — Instale as dependências**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn shap jupyter
```

**Passo 4 — Confirme que o `data.csv` está na mesma pasta do notebook**

O caminho é lido como relativo (`DATA_PATH = "data.csv"`), então o arquivo precisa estar na raiz do repositório, junto ao `.ipynb`.

**Passo 5 — Abra e execute o notebook**
```bash
jupyter notebook tech_challenge_fase1.ipynb
```
Ou abra a pasta no VS Code com a extensão *Jupyter* instalada e clique em **Run All**.

> ⚠️ Como o notebook roda localmente (fora do Colab), a lógica de upload manual não é ativada — o `data.csv` só precisa estar no lugar certo.

---

## 🗂️ Conteúdo do notebook

1. **Dataset** — origem e carregamento dos dados
2. **Carga dos Dados** — leitura do CSV com fallback de upload no Colab
3. **Exploração de Dados (EDA)** — `info()`, `describe()`, checagem de valores ausentes, balanceamento de classes, distribuições e boxplots por diagnóstico
4. **Pré-processamento** — padronização das features numéricas via `StandardScaler`, organizado em `ColumnTransformer` (pronto para extensão a variáveis categóricas futuras)
5. **Análise de Correlação** — matriz de correlação entre features e correlação de cada feature com o diagnóstico
6. **Modelagem** — 4 algoritmos de classificação, todos em `Pipeline` (pré-processamento + modelo) para evitar vazamento de dados entre treino e teste:
   - Regressão Logística (baseline interpretável)
   - Árvore de Decisão
   - Random Forest (300 estimadores)
   - KNN (k escolhido por análise da taxa de erro)
7. **Treinamento e Avaliação** — accuracy, precision, recall e F1 (focadas na classe maligna, a mais crítica clinicamente), validação cruzada estratificada (5 folds), matrizes de confusão, relatório de classificação do melhor modelo e curvas ROC comparativas
8. **Explicabilidade** — feature importance nativa do Random Forest e explicação global via SHAP (`TreeExplainer`)

## 📊 Principais resultados

**Validação cruzada estratificada (5 folds) — Recall macro:**

| Modelo | Recall macro (CV) |
|---|---|
| Logistic Regression | 0.968 ± 0.023 |
| KNN | 0.959 ± 0.023 |
| Random Forest | 0.948 ± 0.019 |
| Decision Tree | 0.918 ± 0.033 |

**Melhor modelo (por recall da classe maligna) — Regressão Logística:**

| Classe | Precision | Recall | F1-score |
|---|---|---|---|
| Maligno | 0.98 | 0.98 | 0.98 |
| Benigno | 0.99 | 0.99 | 0.99 |
| **Acurácia geral** | | | **0.98** |

O foco em **recall da classe maligna** é intencional: em um contexto de diagnóstico, um falso negativo (classificar um tumor maligno como benigno) é o erro mais custoso.

## 🔍 Explicabilidade

As features com maior poder preditivo — tanto pela correlação quanto pela feature importance do Random Forest e pelo SHAP — são majoritariamente as relacionadas a `radius`, `concavity`, `perimeter` e `area`, consistente com a literatura clínica: tumores malignos tendem a ser maiores e apresentar bordas mais irregulares (maior concavidade).

## 🛠️ Tecnologias utilizadas

- Python 3.11
- pandas, numpy
- scikit-learn (pipelines, modelos, métricas, validação cruzada)
- matplotlib, seaborn (visualizações)
- SHAP (explicabilidade)

## Licença
Este projeto foi desenvolvido para fins acadêmicos como parte do Tech Challenge — Fase 1 (FIAP Pós Tech). Uso livre para fins educacionais.
