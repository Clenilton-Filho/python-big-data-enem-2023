import streamlit as st

# Configuração da página
st.set_page_config(
    page_title="Análise ENEM 2023",
    page_icon="🎓",
    layout="wide"
)

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
st.title("🎓 ENEM 2023 - Análise de Dados e Predição de Média")
st.markdown("""
> Projeto desenvolvido para a disciplina de **Tópicos de Big Data em Python**.
""")

st.divider()

st.subheader("⚡ Como usar o aplicativo")
st.markdown("""
1.  Acesse o menu lateral à esquerda.
2.  Escolha **🔮 Simulador** para testar a Inteligência Artificial.
3.  Preencha seus dados socioeconômicos e veja a estimativa da nota.
""")
st.caption("Obs: Os seus dados não serão armazenados ou utilizados para outros fins.")

st.divider()

# Colunas para separar o conteúdo
col1, col2 = st.columns([2, 1], gap="large")

# Informações gerais
with col1:
    st.header("📌 Sobre o Projeto")
    st.write("""
    Este projeto investiga como a **desigualdade social** influencia o desempenho no Exame Nacional do Ensino Médio (ENEM). 
    
    Utilizando microdados públicos do INEP (com ~4 milhões de registros), desenvolvemos um pipeline de Ciência de Dados para:
    1.  **Tratar e limpar** uma base massiva de dados.
    2.  **Analisar** correlações socioeconômicas.
    3.  **Modelar** um sistema de Inteligência Artificial capaz de prever a nota média.
    """)

    st.info("""
    **Objetivo:** Criar um Simulador capaz de predizer o desempenho de um usuário 
    com base em apenas **10 indicadores socioeconômicos**.
    """)

    st.divider()
    st.subheader("🧩 Conclusão:")
    st.write("""
    É possível concluir que, utilizando apenas os dados socioeconômicos capturados, pode-se explicar quase **40% da variação** entre as notas médias de +2 milhões de participantes.
    
    Isso indica que **fatores sociais que não estão sob controle do participante** são determinantes relevantes para o desempenho no ENEM.
    """)

# Informações técnicas
with col2:
    st.header("📊 Performance")
    # Métricas do seu README
    st.metric(label="Acurácia do Modelo", value="89.08%", help="Métrica derivada do MAPE (1 - Erro Percentual Absoluto Médio)")
    st.metric(label="Erro Médio (MAE)", value="57.04 pontos")
    st.metric(label="Variância Explicada (R²)", value="37.06%")
    st.caption("Modelo utilizado: LightGBM")