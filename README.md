# ENEM - Análise de dados socioeconômicos e predição de média do usuário 

#### Este repositório contém scripts para tratamento, treinamento de modelos, geração de gráficos e previsão de média geral do usuário com os dados socioeconômicos do ENEM 2023.

## Equipe Responsável

- Misael - Tratamento de dados
- Victor - Análise de Dados
- Clenilton - Modelos de Machine Learning
- Solário - Coleta de inputs e Apresentação da Média 
- Kairos - Documentação

### Pré-requisitos
- Python 3.8+

1) Criar e ativar um ambiente virtual (recomendado)

```
python -m venv .venv
.venv\Scripts\activate
```

2) Instalar dependências usando requirements.txt

```
pip install -r requirements.txt
```

### Análise de Dados

graficos.ipynb

- Contém um jupyter notebook para gerar gráficos relevantes

### Scripts .py em ordem de execução

filtragem_inicial.py

- Realiza a limpeza e filtragem inicial dos dados

treino_modelos.py

- Treina os modelos e salva em 'modelos/'

inputs.py

- Script que prediz uma nota média geral do usuário para o ENEM 2023 pedindo dados socioeconômicos