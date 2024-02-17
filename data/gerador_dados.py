from faker import Faker
import random 
import pandas as pd

fake = Faker()

# Função para gerar base de dados fictícia.
def dado_fake(usuarios=100, itens=30, avaliacoes=200):
    dados_usuarios = {'usuario_id': [], 'itens_id': [], 'avaliacoes': []}

    for _ in range(avaliacoes):
        dados_usuarios['usuario_id'].append(fake.random_int(min=1, max=usuarios))
        dados_usuarios['itens_id']. append(fake.random_int(min=1, max=itens))
        dados_usuarios['usuario_id'].append(random.randint(1 ,5))


    return pd.DataFrame(dados_usuarios)
