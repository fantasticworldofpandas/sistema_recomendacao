from data.gerador_dados import dado_fake
from utils.preprocessamento import limpeza_dados
from model.recomendacao_model import treinar_modelo, fazer_recomendacoes

# gerar dados fictícios
dados = dado_fake()

# pré-processamento
dados = limpeza_dados(dados)

# treinamento do modelo
modelo = treinar_modelo(dados)

# interação com usuario
usuario_id = int(input('Digite o Id do usuario: '))
num_recomendacoes = int(input('Digite a quantidade de recomendacao: '))

# obter e imprimir recomendações
recomendacoes = fazer_recomendacoes(modelo, usuario_id, dados, num_recomendacoes)
print(f"Recomendações do usuário {usuario_id}: {recomendacoes}")