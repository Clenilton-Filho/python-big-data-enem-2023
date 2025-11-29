import streamlit as st
import pandas as pd
import joblib

# Configuração da página
st.set_page_config(page_title="Simulador de Nota", page_icon="🔮")

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

# Link para o repositório na barra lateral
st.sidebar.markdown("---")
st.sidebar.link_button("Repositório no GitHub ↗", "https://github.com/Clenilton-Filho/python-big-data-enem-2023/tree/main")


# Cabeçalho
st.title("🔮 Simulador de Nota Média")
st.markdown("Responda às perguntas abaixo para que a IA estime sua nota média geral no ENEM.")
st.caption("Obs: Os seus dados não serão armazenados ou utilizados para outros fins.")

# Carregando os modelos (com cache)
@st.cache_resource
def carregar_arquivos():
        
    modelo = joblib.load('./modelos/lightgbm_enem_2023.joblib')
    features = joblib.load('./modelos/features_colunas.joblib')
    return modelo, features

modelo, features_colunas = carregar_arquivos()

# Form para as entradas
with st.form("form_simulador"):
    st.subheader("Dados Socioeconômicos")
    
    # Dividindo em duas colunas
    col1, col2 = st.columns(2)
    
    with col1:

        # Estados
        lista_ufs = ['AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 
                     'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 
                     'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO']
        
        # Default para o que já vem selecionado (5 = CE)
        est_input = st.selectbox("Estado (UF)", lista_ufs, index=5)
        
        regiao_input = st.selectbox("Região do Município", 
                                  ['Capital', 'Metropolitana', 'Interior'])
        
        tipo_escola_input = st.selectbox("Tipo de Escola", 
                                       ['Federal', 'Estadual', 'Privada', 'Não Informado'])
        
        lingua_input = st.selectbox("Língua Estrangeira", ['Inglês', 'Espanhol'])
        
        possui_pc_input = st.radio("Possui computador em casa?", ['Sim', 'Não'], horizontal=True)

    # Coluna à direita
    with col2:
        renda_input = st.selectbox("Renda Familiar",
                                 ['Muito Baixa', 'Baixa', 'Média-Baixa', 'Média', 
                                  'Média-Alta', 'Alta', 'Muito Alta'])
        
        raca_input = st.selectbox("Cor/Raça",
                                ['Branca', 'Preta', 'Parda', 'Amarela', 'Indigena', 'Nao declarado'])
        
        escolaridade_mae_input = st.selectbox("Escolaridade da Mãe",
                                            ['Nunca estudou', 'Fundamental Incompleto', 'Fundamental Completo',
                                             'Ensino Médio', 'Ensino Superior', 'Pós-graduação', 'Não Informado'])
        
        faixa_etaria_input = st.selectbox("Faixa Etária",
                                        ['17-', '18', '19-20', '21', '22-25', '26+'])
        
        acesso_net_input = st.radio("Acesso à Internet?", ['Sim', 'Não'], horizontal=True)

    # Botão de envio no centro
    submitted = st.form_submit_button("Calcular Previsão 🚀", type="primary", use_container_width=True)


if submitted:

    # Criando um DF com os dados brutos
    dados_entrada = pd.DataFrame({
        'sigla_estado': [est_input],
        'raca': [raca_input],
        'lingua': [lingua_input],
        'categoria_renda': [renda_input],
        'escolaridade_mae': [escolaridade_mae_input],
        'possui_computador': [possui_pc_input],
        'tipo_escola': [tipo_escola_input],
        'tipo_regiao': [regiao_input],
        'faixa_etaria': [faixa_etaria_input],
        'acesso_internet': [acesso_net_input]
    })

    # Feedback de carregamento
    with st.spinner('Processando dados...'):

        # Codificando as entradas para o modelo
        input_enc = pd.get_dummies(dados_entrada)
        
        # Reindexando usando as colunas salvas
        input_final = input_enc.reindex(columns=features_colunas, fill_value=0)
        
        # Predição
        predicao = modelo.predict(input_final)[0]

    # Resultado
    st.markdown("---")
    
    st.subheader("Resultado da Análise")
    
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:

        # Exibindo a nota
        st.metric(label="Nota Média Estimada", value=f"{predicao:.0f} pontos")
    
    with col_res2:

        # Mensagens de disclaimer
        st.info(
            """
            ℹ️ **Interpretação:** Essa nota é uma projeção baseada na média de candidatos com perfil 
            socioeconômico similar ao dos dados informados.
            """
        )
        
        st.warning(
            """
            ⚠️ **Importante:** Este aplicativo demonstra como **fatores sociais influenciam o desempenho**, 
            mas ele **não define sua capacidade individual**. 
            O esforço pessoal e o estudo podem superar as estatísticas apresentadas aqui.
            """
        )