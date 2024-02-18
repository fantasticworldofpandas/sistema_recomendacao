def limpeza_dados(dados):
    return dados.drop_duplicates(['usuario_id', 'item_id'])