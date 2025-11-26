# ENEM - Análise de dados socioeconômicos e predição de média do usuário 


#### Este repositório contém scripts para tratamento, treinamento de modelos, geração de gráficos e previsão de média geral do usuário com os dados socioeconômicos do ENEM 2023.


## Equipe Responsável

- Misael - Tratamento de dados
- Victor - Análise de Dados
- Clenilton - Modelos de Machine Learning
- Solário - Coleta de inputs e Apresentação da Média 
- Kairos - Documentação(relatório geral)
  

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


### Execução do script de predição

```
python scripts/inputs.py
```


### Bases de Dados

./dados/enem_2023_limpo.zip

- Base de dados principal gerada pelo tratamento dos microdados do enem 2023, retirada do portal do INEP.

./dados/municipios_rm_limpo.csv

- Base de dados com nomes e códigos de municípios que compõem regiões metropolitanas, gerada pelo tratamento do arquivo 'Recortes Metropolitanos e Aglomerações Urbanas' do portal do IBGE.

### Análise de Dados

./notebooks/grafico_base_atual.ipynb

- Notebook para gerar gráficos relevantes, incluindo média por categoria de renda.


./notebooks/teste_elbow_kmeans.ipynb

- Notebook usado para gerar um gráfico do teste do método elbow para os dados.


### Scripts .py em ordem de execução

./scripts/filtragem_inicial.py

- Realiza a limpeza e filtragem dos dados.

./scripts/treino_modelos.py

- Treina os modelos e salva em 'modelos/'

./scripts/inputs.py

- Script que prediz uma nota média geral do usuário para o ENEM 2023 pedindo somente dados socioeconômicos.
