from faker import Faker
import random 
import pandas as pd

fake = Faker()

# Função para gerar base de dados fictícia.
def dado_fake(usuarios=100, itens=30, avaliacoes=200):
    dados_usuarios = {'usuario_id': [], 'item_id': [], 'avaliacao': []}

    for _ in range(avaliacoes):
        usuario_id = fake.random_int(min=1, max=usuarios)
        item_id = fake.random_int(min=1, max=itens)
        avaliacao = random.randint(1, 5)

        dados_usuarios['usuario_id'].append(usuario_id)
        dados_usuarios['item_id'].append(item_id)
        dados_usuarios['avaliacao'].append(avaliacao)

    return pd.DataFrame(dados_usuarios)
