import pandas as pd 
import numpy as np 

# --- Constantes ---
# Para a filtragem de participantes
PRESENCA_PRESENTE = 1
STATUS_REDACAO_OK = 1
TREINEIRO_NAO = 0
CONCLUSAO_CONCLUIDO = 1
CONCLUSAO_CONCLUINDO_EM_2023 = 2

# Para os mapeamentos
VALOR_PADRAO_NAO_INFORMADO = 'Não Informado'

# --- Funções Auxiliares ---
def mapear_e_preencher(series: pd.Series, mapa: dict) -> pd.Series:
    """
    Aplica um mapeamento a uma Series do pandas e preenche valores
    ausentes com um valor padrão.
    """
    return series.map(mapa).fillna(VALOR_PADRAO_NAO_INFORMADO)

# Carregando o csv com a lista de municípios em regiões metropolitanas
df_rm = pd.read_csv('./dados/municipios_rm_limpo.csv')
municipios_em_rm = set(pd.to_numeric(df_rm['codigo_ibge'], errors='coerce').dropna().astype(int))

# --- filtragem ---
# colunas relevantes
colunas = [
    # Colunas para filtro
    'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'TP_PRESENCA_MT',
    'IN_TREINEIRO', 'TP_STATUS_REDACAO', 'TP_ST_CONCLUSAO',

    # Colunas para o target do modelo
    'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO',

    # Colunas da escola
    'TP_DEPENDENCIA_ADM_ESC', 'TP_LOCALIZACAO_ESC', 'SG_UF_ESC', 'CO_MUNICIPIO_ESC',
    'SG_UF_PROVA', 'CO_MUNICIPIO_PROVA', 'TP_ESCOLA',

    # Colunas do participante
    'TP_LINGUA','TP_FAIXA_ETARIA', 'TP_COR_RACA', 'Q006', 'Q002', 'Q024','Q025'
]

df = pd.read_csv('./dados/enem_2023.csv', usecols=colunas, encoding='latin-1',
                 sep=';', decimal=',', low_memory=False)

# Um DataFrame novo é feito a partir da aplicação de filtros no original
# usando .copy() para evitar warnings
df_filtrado = df[

    # Somente participantes presentes em todas as provas para evitar outliers
    (df['TP_PRESENCA_CN'] == PRESENCA_PRESENTE) & (df['TP_PRESENCA_CH'] == PRESENCA_PRESENTE) &
    (df['TP_PRESENCA_LC'] == PRESENCA_PRESENTE) & (df['TP_PRESENCA_MT'] == PRESENCA_PRESENTE) &

    # Somente redações com nenhum problema, também para evitar outliers
    (df['TP_STATUS_REDACAO'] == STATUS_REDACAO_OK) &

    # Somente quem já concluiu ou estava concluindo o ensino médio naquele ano
    (df['IN_TREINEIRO'] == TREINEIRO_NAO) &
    (df['TP_ST_CONCLUSAO'].isin([CONCLUSAO_CONCLUIDO, CONCLUSAO_CONCLUINDO_EM_2023]))
].copy()
print(f"DataFrame após filtros de participantes: {df_filtrado.shape[0]} linhas.")

# --- Preenchimento ---
# se não houver informação de localização da escola do participante, usa a da prova
df_filtrado['sigla_estado'] = df_filtrado['SG_UF_ESC'].fillna(df_filtrado['SG_UF_PROVA'])

# O mesmo, agora com o município, além de garantir que seja um código numérico
df_filtrado['municipio'] = pd.to_numeric(df_filtrado['CO_MUNICIPIO_ESC'].fillna(df_filtrado['CO_MUNICIPIO_PROVA']), errors='coerce').astype('Int64')

# Lista de capitais para a coluna de região do município
capitais_ibge = {
    2800308, 1501402, 3106200, 1400100, 5300108, 5002704, 5103403, 4106902, 4205407, 2304400, 5208707,
    2507507, 1600303, 2704302, 1302603, 2408102, 1721000, 4314902, 1100205, 2611606, 1200401, 3304557,
    2927408, 2111300, 3550308, 2211001, 3205309
}

def classificar_regiao(municipio: int) -> str:
    """Classifica um município em Capital, Metropolitana ou Interior."""
    if pd.isna(municipio):
        return VALOR_PADRAO_NAO_INFORMADO
    if municipio in capitais_ibge:
        return 'Capital'
    elif municipio in municipios_em_rm:
        return 'Metropolitana'
    else:
        return 'Interior'

df_filtrado['tipo_regiao'] = df_filtrado['municipio'].apply(classificar_regiao)

# --- Mapeamentos para as outras features ---

# Dicionários de mapeamento
mapa_escola_adm = {1: 'Federal', 2: 'Estadual', 3: 'Pública', 4: 'Privada'}
mapa_escola_tipo = {2: 'Pública', 3: 'Privada'}
mapa_lingua = {0: 'Inglês', 1: 'Espanhol'}
mapa_renda = {
    'A': 'Muito Baixa', 'B': 'Muito Baixa', 'C': 'Baixa', 'D': 'Baixa',
    'E': 'Média-Baixa', 'F': 'Média-Baixa', 'G': 'Média', 'H': 'Média',
    'I': 'Média-Alta', 'J': 'Média-Alta', 'K': 'Alta', 'L': 'Alta',
    'M': 'Alta', 'N': 'Alta', 'O': 'Muito Alta', 'P': 'Muito Alta', 'Q': 'Muito Alta'
}
mapa_raca = {0: 'Nao declarado', 1: 'Branca', 2: 'Preta', 3: 'Parda', 4: 'Amarela', 5: 'Indigena'}
mapa_escolaridade = {
    'A': 'Nunca estudou', 'B': 'Fundamental Incompleto', 'C': 'Fundamental Incompleto',
    'D': 'Fundamental Completo', 'E': 'Ensino Médio', 'F': 'Ensino Superior',
    'G': 'Pós-graduação', 'H': 'Não Informado'
}
mapa_computador = {'A': 'Não', 'B': 'Sim', 'C': 'Sim', 'D': 'Sim', 'E': 'Sim'}
mapa_internet = {'A': 'Não', 'B': 'Sim'}

# Aplicando os mapeamentos
df_filtrado['tipo_escola'] = mapear_e_preencher(df_filtrado['TP_DEPENDENCIA_ADM_ESC'], mapa_escola_adm)
df_filtrado['tipo_escola'] = df_filtrado['tipo_escola'].fillna(mapear_e_preencher(df_filtrado['TP_ESCOLA'], mapa_escola_tipo))
df_filtrado['lingua'] = mapear_e_preencher(df_filtrado['TP_LINGUA'], mapa_lingua)
df_filtrado['categoria_renda'] = mapear_e_preencher(df_filtrado['Q006'], mapa_renda)
df_filtrado['raca'] = mapear_e_preencher(df_filtrado['TP_COR_RACA'], mapa_raca)
df_filtrado['escolaridade_mae'] = mapear_e_preencher(df_filtrado['Q002'], mapa_escolaridade)
df_filtrado['possui_computador'] = mapear_e_preencher(df_filtrado['Q024'], mapa_computador)
df_filtrado['acesso_internet'] = mapear_e_preencher(df_filtrado['Q025'], mapa_internet)

# Lista com condições para usar com numpy,
# nesse caso para mapear faixas etárias
condicoes_idade = [
    (df_filtrado['TP_FAIXA_ETARIA'] <= 2), # Menor de 17 ou 17 anos
    (df_filtrado['TP_FAIXA_ETARIA'] == 3), # 18 anos 
    (df_filtrado['TP_FAIXA_ETARIA'].between(4, 5)), # 19 a 20 anos
    (df_filtrado['TP_FAIXA_ETARIA'] == 6), # 21 anos 
    (df_filtrado['TP_FAIXA_ETARIA'].between(7, 10)), # 22 a 25 anos 
    (df_filtrado['TP_FAIXA_ETARIA'] > 10) # 26 anos ou mais 
]

# Categorias selecionadas devido à relação com a média geral analisada anteriormente
categorias_idade = ['17-', '18', '19-20', '21', '22-25', '26+']
df_filtrado['faixa_etaria'] = np.select(condicoes_idade, categorias_idade, default='Não Informado')

# Garantindo dados numéricos para as notas
nota_cols = ['NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_MT','NU_NOTA_REDACAO']
for c in nota_cols:
    df_filtrado[c] = pd.to_numeric(df_filtrado[c], errors='coerce')

# Removendo nulos que tenham ficado
df_filtrado = df_filtrado.dropna(subset=nota_cols)

# Criando a coluna nota_media
df_filtrado['nota_media'] = df_filtrado[nota_cols].mean(axis=1)

# Dropando as colunas de filtro e as temporárias que já foram usadas
colunas_drop = [
    'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'TP_PRESENCA_MT',
    'IN_TREINEIRO', 'TP_STATUS_REDACAO', 'TP_ST_CONCLUSAO',
    'SG_UF_PROVA', 'CO_MUNICIPIO_PROVA', 'TP_DEPENDENCIA_ADM_ESC', 'TP_ESCOLA',
    'TP_LOCALIZACAO_ESC', 'TP_COR_RACA', 'TP_FAIXA_ETARIA', 'Q006', 'Q002', 'Q024','Q025',
    'TP_LINGUA', 'SG_UF_ESC', 'CO_MUNICIPIO_ESC', 'municipio'
]
df_final = df_filtrado.drop(columns=colunas_drop)

# Garantindo que as colunas com dados categóricos têm tipo correto
colunas_para_categoria = [
    'sigla_estado', 'raca', 'lingua',
    'categoria_renda', 'escolaridade_mae', 'possui_computador',
    'tipo_escola', 'tipo_regiao', 'faixa_etaria', 'acesso_internet'
]
for c in colunas_para_categoria:
    if c in df_final.columns:
        df_final[c] = df_final[c].astype('category')

# Salvando o resultado final
df_final.to_csv('./dados/enem_2023_limpo.csv', index=False)

# --- Informações do csv final ---
print("\n--- Informações do DataFrame final ---")
df_final.info()
print("\n--- Exemplo dos dados ---")
print(df_final.head())
print(f"\n Base final salva com {df_final.shape[0]} registros e {df_final.shape[1]} colunas.")