import pandas as pd
import numpy as np

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
    (df['TP_PRESENCA_CN'] == 1) & (df['TP_PRESENCA_CH'] == 1) &
    (df['TP_PRESENCA_LC'] == 1) & (df['TP_PRESENCA_MT'] == 1) &

    # Somente redações com nenhum problema, também para evitar outliers, que nesse caso seriam notas 0
    (df['TP_STATUS_REDACAO'] == 1) &

    # Somente quem já concluiu ou estava concluindo o ensino médio naquele ano
    # Isso garante que os participantes restantes já tiveram oportunidade de ver os conteúdos cobrados
    (df['IN_TREINEIRO'] == 0) &
    (df['TP_ST_CONCLUSAO'].isin([1, 2]))
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

# Se for da lista de capitais, é mapeada como 'Capital'.
# caso não for, verifica se está na lista de municípios em regiões metropolitanas. Se estiver, 'Metropolitana'
# se não, é mapeada como 'Interior'
def classificar_regiao(municipio):
    # Adiciona verificação para valores nulos
    if pd.isna(municipio):
        return 'Não Informado'
    if municipio in capitais_ibge:
        return 'Capital'
    elif municipio in municipios_em_rm:
        return 'Metropolitana'
    else:
        return 'Interior'

# .apply() executa a função para cada município, criando a nova coluna de tipo de região.
df_filtrado['tipo_regiao'] = df_filtrado['municipio'].apply(classificar_regiao)

# --- Mapeamentos para as outras features ---

# Usamos a coluna de dependência administrativa como padrão para o tipo de escola
# Municipal mapeada como pública, pois tem poucos dados 
df_filtrado['tipo_escola'] = ( 
    df_filtrado['TP_DEPENDENCIA_ADM_ESC'].map( 
        {1: 'Federal', 2: 'Estadual', 3: 'Pública', 4: 'Privada'})) 
 
# A coluna de tipo de escola vai ser usada caso não tenha a outra informação
df_filtrado['tipo_escola'] = ( 
    df_filtrado['tipo_escola'].fillna( 
        df_filtrado['TP_ESCOLA'].map( 
            {2: 'Pública', 3: 'Privada'}))) 
df_filtrado['tipo_escola'] = df_filtrado['tipo_escola'].fillna('Não Informado')

# Coluna de linguagem estrangeira
df_filtrado['lingua'] = ( 
    df_filtrado['TP_LINGUA'].map(  
        {0: 'Inglês', 1: 'Espanhol'} 
    ).fillna('Não Informado')) 

# Coluna de categoria de renda, agrupando categorias
df_filtrado['categoria_renda'] = ( 
    df_filtrado['Q006'].map({ 
        'A': 'Muito Baixa', 'B': 'Muito Baixa',     # Nenhuma Renda até 1 SM
        'C': 'Baixa', 'D': 'Baixa',                 # > 1 SM até 2 SM
        'E': 'Média-Baixa', 'F': 'Média-Baixa',     # > 2 SM até 3 SM
        'G': 'Média', 'H': 'Média',                 # > 3 SM até 5 SM
        'I': 'Média-Alta', 'J': 'Média-Alta',       # > 5 SM até 10 SM
        'K': 'Alta', 'L': 'Alta', 'M': 'Alta', 'N': 'Alta', # > 10 SM até 20 SM
        'O': 'Muito Alta', 'P': 'Muito Alta', 'Q': 'Muito Alta' # > 20 SM
    }).fillna('Não informado')) 

# Coluna de declaração de raça
df_filtrado['raca'] = ( 
    df_filtrado['TP_COR_RACA'].map( 
        { 0: 'Nao declarado', 1: 'Branca', 2: 'Preta',  
          3: 'Parda', 4: 'Amarela', 5: 'Indigena' } 
    ).fillna('Nao declarado')) 
 
# Coluna de nível de escolaridade da mãe, agrupando algumas categorias
df_filtrado['escolaridade_mae'] = (
    df_filtrado['Q002'].map( 
        { 'A': 'Nunca estudou', 'B': 'Fundamental Incompleto', 
        'C': 'Fundamental Incompleto', 'D': 'Fundamental Completo', 'E': 'Ensino Médio',
        'F': 'Ensino Superior', 'G': 'Pós-graduação', 'H': 'Não Informado' } 
    ).fillna('Não Informado')) 

# Coluna de posse de computador em casa 
df_filtrado['possui_computador'] = ( 
    df_filtrado['Q024'].map( 
        {'A': 'Não', 'B': 'Sim', 'C': 'Sim', 'D': 'Sim', 'E': 'Sim'} 
    ).fillna('Não Informado')) 
 
# Coluna de acesso à internet em casa
df_filtrado['acesso_internet'] = ( 
    df_filtrado['Q025'].map( 
        {'A': 'Não', 'B': 'Sim'} 
    ).fillna('Não Informado'))

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