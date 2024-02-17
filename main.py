from data.gerador_dados import gerar_dados_falsos
from utils.preprocessamento import limpar_dados
from model.recomendacao_model import treinar_modelo

# gerar dados fictícios
dados = gerar_dados_falsos()

# pré-processamento
dados = limpar_dados(dados)

# treinamento do modelo
modelo, previsoes = treinar_modelo(dados)