from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
import pandas as pd

def treinar_modelo(dados):
    # dividir o conjunto de dados em treino e teste
    train_data, test_data = train_test_split(dados, test_size=0.2, random_state=42)

    # criar matrizes de usuário-item
    matriz_usuario_item = train_data.pivot(index='usuario_id', columns='item_id', values='avaliacao').fillna(0)

    # aplicar a decomposição da matriz usando TruncatedSVD
    modelo = TruncatedSVD(n_components=10, random_state=42)
    modelo.fit(matriz_usuario_item)

    # fazer previsões para o conjunto de teste
    predicoes = modelo.inverse_transform(modelo.transform(test_data.pivot(index='usuario_id', columns='item_id', values='avaliacao').fillna(0)))

    # avaliar o modelo
    mse = mean_squared_error(test_data['avaliacao'], predicoes.flatten())
    print(f'Erro medio: {mse}')

    return modelo