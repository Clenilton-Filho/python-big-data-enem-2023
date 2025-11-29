import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Dashboard com gráficos", page_icon="📈", layout="wide")

# Cabeçalho
st.title("📈 Análise Exploratória")
st.markdown("Explore os dados através de gráficos baseados no dataset completo.")

# Carrega a base de dados em cache (~25MB)
@st.cache_data
def carregar_dados():
    return pd.read_parquet("dados/enem_2023_limpo.parquet")

df = carregar_dados()

# Filtro por estados na barra lateral
uf_selecionada = st.sidebar.multiselect(
    "🔎 Filtrar por Estado",

    # Em ordem alfabética
    options=sorted(df['sigla_estado'].unique()),
    default=sorted(df['sigla_estado'].unique())
)

# Filtrando
df_filtered = df[df['sigla_estado'].isin(uf_selecionada)]

st.sidebar.markdown("---")

# Créditos na barra lateral
st.sidebar.markdown("### 👥 Autores")
st.sidebar.markdown("""
- **Clenilton Vasconcelos**
- **Misael Alves**
- **Solário Bringel**
- **Francisco Vitor**
- **Kairos Dimarães**
""")

st.sidebar.markdown("---")

# Badges das tecnologias utilizadas
st.sidebar.markdown("### 🛠️ Tecnologias")
st.sidebar.markdown(
    """
    ![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54)
    ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)
    ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
    ![LightGBM](https://img.shields.io/badge/LightGBM-%23150458.svg?style=flat&logo=lightgbm&logoColor=white)
    """
)

# Gráficos

# 3 Abas
tab1, tab2, tab3 = st.tabs(["💰 Renda", "🏫 Infraestrutura", "👨‍👩‍👧 Família & Sociedade"])

# Renda
with tab1:
    st.subheader("Renda Familiar vs Nota Média")
    
    # Agrupando média
    df_renda = df_filtered.groupby('categoria_renda')[['nota_media']].mean().reset_index()
    ordem_renda = ['Muito Baixa', 'Baixa', 'Média-Baixa', 'Média', 'Média-Alta', 'Alta', 'Muito Alta']
    
    fig_renda = px.bar(
        df_renda, 
        x='categoria_renda', 
        y='nota_media',
        color='nota_media',
        category_orders={'categoria_renda': ordem_renda},
        title="Média de Notas por Classe Social",
        color_continuous_scale='Bluyl',
        text_auto='.0f' 
    )
    fig_renda.update_layout(xaxis_title=None, yaxis_title="Nota Média")
    st.plotly_chart(fig_renda, use_container_width=True)

    st.info("💡 **Insight:** Como é esperado, a renda familiar ainda é um fator relevante para o desempenho.")

# Infraestrutura
with tab2:

    # 2 colunas de gráficos
    col1, col2 = st.columns(2)
    
    with col1:

        # Tipos de Escola
        st.subheader("Tipos de Escola")
        
        # Barras de Média
        df_escola = df_filtered.groupby('tipo_escola')[['nota_media']].mean().reset_index()
        
        # Removendo a categoria "Pública" para evitar redundância com Estadual/Federal
        df_escola = df_escola[df_escola['tipo_escola'] != 'Pública']

        fig_escola = px.bar(
            df_escola, 
            x='tipo_escola', 
            y='nota_media', 
            color='tipo_escola',
            title="Média por Tipo de Escola",
            text_auto='.0f',
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_escola.update_layout(showlegend=False, xaxis_title=None)
        st.plotly_chart(fig_escola, use_container_width=True)
            
    with col2:

        # Computador em casa
        st.subheader("Computador em casa")
        
        # Histograma com sobreposição
        fig_pc = px.histogram(
            df_filtered,
            x='nota_media',
            color='possui_computador',
            title="Distribuição de Médias",
            barmode='overlay',

            # Com transparência
            opacity=0.7,
            nbins=40,
            color_discrete_sequence=['#EF553B', '#636EFA']
        )
        fig_pc.update_layout(xaxis_title="Nota", yaxis_title="Qtd. Alunos")
        st.plotly_chart(fig_pc, use_container_width=True)

    st.divider()

    # Acesso à internet
    st.subheader("Acesso à Internet")
    col_net1, col_net2 = st.columns([1, 2])
    
    with col_net1:
        df_net = df_filtered['acesso_internet'].value_counts().reset_index()
        df_net.columns = ['Tem Internet', 'Quantidade']
        fig_net = px.pie(
            df_net, 
            values='Quantidade', 
            names='Tem Internet', 
            title='Participantes: Com Internet vs Sem Internet',
            hole=0.5
        )
        st.plotly_chart(fig_net, use_container_width=True)
        
    with col_net2:
        df_media_net = df_filtered.groupby('acesso_internet')[['nota_media']].mean().reset_index()
        
        # Gráfico de barras
        fig_comp = px.bar(
            df_media_net,
            y='acesso_internet',
            x='nota_media',
            orientation='h',
            title="Média de Notas",
            text_auto='.0f',
            color='acesso_internet',
            color_discrete_map={'Sim': '#1F77B4', 'Não': '#76B4E5'}
        )
        fig_comp.update_layout(xaxis_title="Nota Média", yaxis_title=None, showlegend=False)
        st.plotly_chart(fig_comp, use_container_width=True)

    st.info("💡 **Insight:** A correlação entre infraestrutura e desempenho é evidente nesses aspectos, tendo também relação com a renda familiar.")

# Família e sociedade
with tab3:
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Escolaridade da Mãe")
        df_mae = df_filtered.groupby('escolaridade_mae')[['nota_media']].mean().reset_index()
        ordem_mae = [
            'Nunca estudou', 'Fundamental Incompleto', 'Fundamental Completo',
            'Ensino Médio', 'Ensino Superior', 'Pós-graduação', 'Não Informado'
        ]
        
        fig_mae = px.bar(
            df_mae,
            y='escolaridade_mae',
            x='nota_media',
            orientation='h',
            title="Média por Escolaridade Materna",
            category_orders={'escolaridade_mae': ordem_mae},
            color='nota_media',
            color_continuous_scale='Viridis',
            text_auto='.0f'
        )
        fig_mae.update_layout(xaxis_title="Nota Média", yaxis_title=None)
        st.plotly_chart(fig_mae, use_container_width=True)
        
    with col_b:

        # Média por raça/cor
        st.subheader("Média por Raça/Cor")
        
        # Barras ordenadas
        df_raca = df_filtered.groupby('raca')[['nota_media']].mean().reset_index().sort_values('nota_media')
        
        fig_raca = px.bar(
            df_raca,
            x='raca',
            y='nota_media',
            color='nota_media',
            title="Média de Notas por Autodeclaração de Raça",
            text_auto='.0f',
            color_continuous_scale='Sunsetdark'
        )
        fig_raca.update_layout(xaxis_title=None, yaxis_title="Nota Média")
        st.plotly_chart(fig_raca, use_container_width=True)

    st.info("💡 **Insight:** A barra de escolaridade da mãe cresce quase perfeitamente alinhada com a nota, indicando forte correlação.")