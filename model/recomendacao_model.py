from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import precision_score

def treinar_modelo(dados):
    # dividir o conjunto de dados em treino e teste
    train_data, test_data = train_test_split(dados, test_size=0.2, random_state=42)

    # criar matrizes de usuário-item
    matriz_usuario_item_train = train_data.pivot(index='usuario_id', columns='item_id', values='avaliacao').fillna(0)
    matriz_usuario_item_test = test_data.pivot(index='usuario_id', columns='item_id', values='avaliacao').fillna(0)

    # Garantir que ambas as matrizes tenham as mesmas colunas
    cols_diff = set(matriz_usuario_item_train.columns) - set(matriz_usuario_item_test.columns)
    for col in cols_diff:
        matriz_usuario_item_test[col] = 0

    # aplicar a decomposição da matriz usando TruncatedSVD
    modelo = TruncatedSVD(n_components=30, random_state=42)
    modelo.fit(matriz_usuario_item_train)

    # fazer previsões para o conjunto de teste
    predicoes = modelo.inverse_transform(modelo.transform(matriz_usuario_item_test))

    # avaliar o modelo
    mse = mean_squared_error(matriz_usuario_item_test.values.flatten(), predicoes.flatten())
    print(f'Erro médio: {mse}')

    # calcular a precisão
    precision = precision_score((matriz_usuario_item_test > 0).values.flatten(), (predicoes > 0).flatten(), average='micro')
    print(f'Precisão: {precision}')

    return modelo


def fazer_recomendacoes(modelo, usuario_id, dados, num_recomendacoes=5):
    n_itens = dados['item_id'].nunique()  

    itens_nao_avaliados = set(range(1, n_itens + 1)) - set(dados[dados['usuario_id'] == usuario_id]['item_id'])
    
    # fazer previsões para itens não avaliados
    predicoes = modelo.predict([(usuario_id, item_id, 0) for item_id in itens_nao_avaliados])
    
    # ordenar as previsões por avaliação prevista em ordem decrescente
    recomendacoes = sorted(predicoes, key=lambda x: x.est, reverse=True)
    
    # retornar os IDs dos itens recomendados
    return [recomendacao.iid for recomendacao in recomendacoes[:num_recomendacoes]]