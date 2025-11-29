# 🎓 Simulador ENEM 2023 (dados socioeconômicos)

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-%23150458.svg?style=for-the-badge&logo=lightgbm&logoColor=white)

> 🚀 **Versão Web Interativa** do projeto de Tópicos de Big Data em Python.

---

## 🔗 Acesse o App
Clique abaixo para acessar a aplicação hospedada no Streamlit Cloud:

[![Acessar Simulador](https://img.shields.io/badge/Acessar_App-Ver_Online-2ea44f?style=for-the-badge&logo=google-chrome)](#)

---

## 📌 Sobre esta Versão
Esta branch (`deploy-streamlit`) contém o código-fonte da **interface gráfica** do projeto. Ela foi otimizada para nuvem, removendo datasets massivos e focando na experiência do usuário.

O sistema utiliza modelos de Machine Learning pré-treinados para estimar a nota média de um candidato no ENEM com base em **10 indicadores socioeconômicos**, evidenciando o impacto da desigualdade social na educação.

> 📂 **Procurando o código de tratamento de dados e treino?** > O pipeline completo de Ciência de Dados (ETL de 1.7GB, EDA e Treinamento) está na branch principal:  
> [**Clique aqui para ver a branch Main**](https://github.com/Clenilton-Filho/python-big-data-enem-2023/tree/main)

---

## ✨ Funcionalidades

* **🔮 Simulador Interativo:** Preencha formulários simples (Estado, Renda, Escola) e receba a predição em tempo real.
* **📈 Dashboard Analytics:** Visualize os gráficos e insights gerados durante a análise exploratória dos dados.
* **🧠 Motor de IA:** Roda um modelo **LightGBM** com 89% de acurácia estimada, capaz de capturar relações não-lineares entre renda e desempenho.

---

## 🛠️ Como rodar localmente

Se você quiser testar esta interface no seu computador:

1.  **Clone o repositório e mude para a branch de deploy:**
    ```bash
    git clone [https://github.com/Clenilton-Filho/python-big-data-enem-2023.git](https://github.com/Clenilton-Filho/python-big-data-enem-2023.git)
    cd python-big-data-enem-2023
    git checkout deploy-streamlit
    ```

2.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Execute o Streamlit:**
    ```bash
    streamlit run Home.py
    ```
    O navegador abrirá automaticamente em `http://localhost:8501`.

---

## 📊 Métricas do Modelo (LightGBM)

O modelo utilizado nesta interface obteve os seguintes resultados na base de teste:

| Métrica | Valor | Descrição |
| :--- | :--- | :--- |
| **Acurácia** | **89.08%** | Derivada do MAPE (1 - Erro Percentual) |
| **MAE** | **57.04** | Erro Médio Absoluto (em pontos) |
| **R²** | **37.06%** | Variação explicada pelos dados socioeconômicos |

> **Nota:** O modelo demonstra a forte influência de fatores sociais, mas não determina a capacidade individual do estudante.

---

## 📂 Estrutura de Arquivos

```text
├── modelos/                 # Arquivos .joblib (IA treinada)
├── pages/                   # Páginas do aplicativo
│   └── 1_🔮_Simulador.py
├── imagens/                 # Assets visuais (gráficos)
├── .streamlit/              # Configurações do streamlit
├── Home.py                  # Página inicial
└── requirements.txt         # Dependências do app
```

---

## 🏛️ Fontes dos dados 

- INEP — Microdados ENEM 2023:
  - https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem
  - Acessado em: 19/11/2025.
  
- IBGE — Recortes Metropolitanos e Aglomerações Urbanas:
  - https://www.ibge.gov.br/geociencias/organizacao-do-territorio/divisao-regional/18354-recortes-metropolitanos-e-aglomeracoes-urbanas.html
  - Acessado em: 19/11/2025.