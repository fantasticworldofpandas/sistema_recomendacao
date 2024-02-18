from data.gerador_dados import dado_fake
from utils.preprocessamento import limpeza_dados
from model.recomendacao_model import treinar_modelo

# gerar dados fictícios
dados = dado_fake()

# pré-processamento
dados = limpeza_dados(dados)

# treinamento do modelo
modelo = treinar_modelo(dados)