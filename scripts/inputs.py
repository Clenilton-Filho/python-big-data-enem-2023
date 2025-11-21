import pandas as pd
import joblib

# Carregando o arquivo final, preparado pelo script de filtragem.
print("--- Carregando arquivo csv ---")
df = pd.read_csv('./dados/enem_2023_limpo.csv')

# Carregando o modelo que vai ser usado
print('\n--- Carregando modelo salvo ---')
modelo = joblib.load('./modelos/lightgbm_enem_2023.joblib')

# Carregando a lista de colunas usadas no treino depois da codificação one-hot
features_colunas = joblib.load('./modelos/features_colunas.joblib')

# Colunas que vão ser usadas
features = [
    'sigla_estado', 'raca', 'lingua', 'categoria_renda',
    'escolaridade_mae', 'possui_computador', 'tipo_escola',
    'tipo_regiao', 'faixa_etaria', 'acesso_internet'
]

# Função que recebe os inputs do usuário e usa o modelo para fazer uma previsão de nota média.
def prever_nota(modelo, sigla_estado, raca, lingua, categoria_renda, escolaridade_mae,
                possui_computador, tipo_escola, tipo_regiao, faixa_etaria, acesso_internet):

    # Criamos um DataFrame de uma linha com os dados
    input_df = pd.DataFrame({
        'sigla_estado': [sigla_estado.strip().upper()],
        'raca': [raca],
        'lingua': [lingua],
        'categoria_renda': [categoria_renda],
        'escolaridade_mae': [escolaridade_mae],
        'possui_computador': [possui_computador],
        'tipo_escola': [tipo_escola],
        'tipo_regiao': [tipo_regiao],
        'faixa_etaria': [faixa_etaria],
        'acesso_internet': [acesso_internet]
    }, columns=features)

    # Aplicamos get_dummies e reindexamos para garantir colunas numéricas para o modelo
    # fill_value=0, já que usamos drop_first no treino
    input_enc = pd.get_dummies(input_df).reindex(columns=features_colunas, fill_value=0)

    # Recebendo a média do modelo usando as respostas já codificadas
    pred = modelo.predict(input_enc)[0]

    # A função retorna a nota média geral que o modelo gerou
    return float(pred)


# Imprimindo o título do programa
print("\n--- Simulador de Nota Média do ENEM via dados socioeconômicos ---\n")
print("Insira os dados para prever sua nota média geral:\n")

# Função para facilitar a coleta dos inputs
def get_user_input(prompt, options):
    print(f"\n{prompt}")
    for i, option in enumerate(options):
        print(f"  {i + 1}. {option}")

    while True:
        choice = input("Escolha o número da sua resposta: ")
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        else:
            print(f"ERRO: Por favor, digite um número entre 1 e {len(options)}.")


# Coletando os 10 inputs
est_input = input("Estado da escola/prova (ex: CE): ").strip().upper()
regiao_input = get_user_input("Sua cidade é:", ['Capital', 'Metropolitana', 'Interior'])
tipo_escola_input = get_user_input("Tipo de escola:", ['Federal', 'Estadual', 'Pública', 'Privada', 'Não Informado'])
lingua_input = get_user_input("Língua estrangeira:", ['Inglês', 'Espanhol'])
renda_input = get_user_input("Renda familiar:",
                             ['Muito Baixa', 'Baixa', 'Média-Baixa', 'Média', 'Média-Alta', 'Alta', 'Muito Alta'])
raca_input = get_user_input("Qual a sua raça?",
                            ['Branca', 'Preta', 'Parda', 'Amarela', 'Indigena', 'Nao declarado'])
escolaridade_mae_input = get_user_input("Escolaridade da mãe:",
                                        ['Nunca estudou', 'Fundamental Incompleto', 'Fundamental Completo',
                                         'Ensino Médio', 'Ensino Superior', 'Pós-graduação', 'Não Informado'])
possui_pc_input = get_user_input("Possui computador em casa?", ['Sim', 'Não'])
acesso_net_input = get_user_input("Possui acesso à internet em casa?", ['Sim', 'Não'])
faixa_etaria_input = get_user_input("Qual sua faixa etária?",
                                    ['17-', '18', '19-20', '21', '22-25', '26+'])

# Chamando a função de predição com todos os parâmetros
nota_prevista = prever_nota(
    modelo, est_input, raca_input, lingua_input, renda_input,
    escolaridade_mae_input, possui_pc_input, tipo_escola_input, regiao_input,
    faixa_etaria_input, acesso_net_input
)

# Imprimindo a nota média geral que o modelo gerou para o usuário
print("\n-------------------------------------------")
print(f"Nota média prevista: {nota_prevista:.2f}")
print("-------------------------------------------")
