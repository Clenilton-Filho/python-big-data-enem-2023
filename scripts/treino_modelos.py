import pandas as pd
import time

# para salvar os modelos
import joblib
import os

# Biblioteca sklearn e seus módulos para treinamento dos modelos usados
# (Linear Regression e RandomForestRegressor)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Modelo não supervisionado que usa K-Means de maneira eficiente
from sklearn.cluster import MiniBatchKMeans

# Módulo de métricas dos modelos
# MAE e R² para os 3 supervisionados
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error, r2_score

# Modelo muito leve que implementa árvore de decisão (mais explicado depois)
# É necessário ter instalado a biblioteca separadamente para usar
from lightgbm import LGBMRegressor

# Carregando a base limpa
print(f"\n--- Carregando o csv ---")

df = pd.read_csv("./dados/enem_2023_limpo.csv")

# Definindo features (características) e target (alvo)
features = [
    'sigla_estado', 'tipo_regiao', 'tipo_escola', 'lingua',
    'categoria_renda', 'raca', 'escolaridade_mae',
    'possui_computador', 'faixa_etaria', 'acesso_internet'
]
target = 'nota_media'

print("\nFeatures usadas:", features)

# Codificando as variáveis de categoria para que os modelos entendam
X = pd.get_dummies(df[features].fillna('NA'), drop_first=True)

# Garantindo que o alvo está em float
y = df[target].astype(float)

# Dividindo dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(

    # Definindo tamanho do teste como 20% e random_state para consistência entre resultados
    X, y, test_size=0.2, random_state=42
)

# --- Modelos ---

# LinearRegression - Modelo básico e rápido que usa linhas para predição
# ele tenta encontrar a melhor linha reta que correlacione as diferentes features
lr = LinearRegression()


# RandomForestRegressor - Modelo que funciona usando um número especificado
# de árvores de decisão que classificam os dados de maneira aleatória por árvore
# cada árvore só pode escolher uma das decisões que fez para entregar ao modelo

# Usando uma configuração que foi testada e aprovada em desempenho e velocidade
rf_otimizado = RandomForestRegressor(
    n_estimators=50,        # número de árvores reduzido, mas eficiente
    max_samples=0.3,        # cada árvore usa 30% dos dados, acelerando o processo
    max_depth=20,           # limite de decisões para evitar sobrecarregamento
    min_samples_leaf=5,     # mínimo de folhas para também evitar sobrecarregamento
    n_jobs=-1,              # usando todos os processadores
    bootstrap=True,         # método para obtenção das amostras
    random_state=42,
)


# LightGBM - Modelo muito leve que aplica árvores de decisão em sequência
# cada árvore vai melhorar o resultado olhando para a árvore anterior
# e corrigindo os possíveis erros dela

# Aumentando alguns parâmetros para tentar melhorar o desempenho
# demora mais, mas pode aumentar o r²
lgbm_precisao = LGBMRegressor(
    n_estimators=500,       # mais árvores
    learning_rate=0.02,      # aprende mais devagar
    max_depth=10,            # árvores com mais níveis
    num_leaves=51,           # mais folhas por árvore
    subsample=0.8,           # usando 80% para amostras em cada uma
    colsample_bytree=0.8,    # 80% das colunas para cada
    random_state=42,
    n_jobs=-1,
    verbose=-1               # -1 para não exibir mensagens durante o treino
)

# --- Treinamento inicial (sem clusters) ---

print("\n--- Treinamento inicial (sem clusters) ---")

# Lista de modelos que serão executados
modelos = [
    ("LinearRegression", lr),
    ("RandomForest", rf_otimizado),
    ("LightGBM", lgbm_precisao)
]

# Loop para treinar e avaliar antes dos clusters
for nome, modelo in modelos:

    # Exibe o nome do modelo
    # e calcula o tempo entre o início e fim do fit (treinamento)
    print(f"\n--- Modelo: {nome} ---")
    inicio = time.time()
    modelo.fit(X_train, y_train)
    tempo_treino = time.time() - inicio

    # Calcula o tempo entre o início e o fim da predição (responder a novos dados)
    inicio_pred = time.time()
    preds = modelo.predict(X_test)
    tempo_pred = time.time() - inicio_pred

    # MAE aqui vai significar a quantidade média de pontos
    mae = mean_absolute_error(y_test, preds)

    # Já o R² vai medir a porcentagem de variação entre as notas que o modelo consegue explicar
    r2 = r2_score(y_test, preds)

    # Calcula a porcentagem de erro nas notas
    mape = mean_absolute_percentage_error(y_test, preds)

    # Transformamos em "Acurácia" subtraindo o MAPE de 1
    acuracia = 1 - mape

    # Printando os resultados
    print(f"Tempo de treino: {tempo_treino:.2f}s | Tempo de predição: {tempo_pred:.2f}s")
    print(f'MAE (erro médio em pontos): {mae:.2f}')
    print(f'Acurácia estimada (derivada do MAPE): {acuracia * 100:.2f}%')
    print(f'R² (variação capturada): {r2 * 100:.2f}%')

# --- Treinamento não supervisionado---

print("\n\n--- Treino com o modelo MiniBatchKMeans (Não supervisionado) ---")

# O modelo MiniBatchKMeans funciona como o K-Means normal,
# Mas de forma mais eficiente para quantidades muito grandes de dados,
# Usando quantidades pequenas de amostras (batches) por vez

# 6 clusters, quantidade testada com o método elbow
# batch_size é a quantidade de dados em cada batch, 256 nesse caso
# n_init=10 para evitar avisos e melhorar a estabilidade
kmeans = MiniBatchKMeans(n_clusters=6, random_state=42, batch_size=256, n_init=10)

# Treina o K-Means só com os dados de treino
kmeans.fit(X_train)

# --- Preparando os dados de treino e teste de novo, com a coluna nova ---

# .copy() para evitar avisos
X_train_clusters = X_train.copy()
X_test_clusters = X_test.copy()

# Predict com os dados de treino
clusters_train = kmeans.predict(X_train)
X_train_clusters['perfil_participante'] = clusters_train

# Agora com os dados de teste
clusters_test = kmeans.predict(X_test)
X_test_clusters['perfil_participante'] = clusters_test

# Salva a lista de colunas para que o script de inputs use
# exatamente as mesmas colunas que o modelo usou, incluindo os clusters
joblib.dump(list(X_train_clusters.columns), os.path.join('./modelos', "features_colunas.joblib"))
print("\nLista de colunas com cluster salva")


# --- Treinamento e resultados com a nova coluna---

print("\n--- Novo treino dos modelos supervisionados, agora com clusters ---")

# Salvando o modelo
joblib.dump(kmeans, os.path.join("./modelos", "minibatchkmeans_enem_2023.joblib"))
print("\nModelo K-Means salvo.")

for nome, modelo in modelos:

    print(f"\n--- Modelo (Com Clusters): {nome} ---")
    inicio = time.time()
    modelo.fit(X_train_clusters, y_train)
    tempo_treino = time.time() - inicio

    inicio_pred = time.time()
    preds = modelo.predict(X_test_clusters)
    tempo_pred = time.time() - inicio_pred

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    acuracia = 1 - mape

    print(f"Tempo de treino: {tempo_treino:.2f}s | Tempo de predição: {tempo_pred:.2f}s")
    print(f'MAE (erro médio em pontos): {mae:.2f}')
    print(f'Acurácia estimada (derivada do MAPE): {acuracia * 100:.2f}%')
    print(f'R² (variação capturada): {r2 * 100:.2f}%')

    # Salvando os modelos com clusters
    nome_arquivo = f"{nome.lower()}_enem_2023.joblib"
    caminho_arquivo = os.path.join("./modelos", nome_arquivo)
    joblib.dump(modelo, caminho_arquivo)

print(f"\nOs modelos foram salvos")