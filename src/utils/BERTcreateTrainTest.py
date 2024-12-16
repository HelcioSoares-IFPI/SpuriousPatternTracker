import pandas as pd
import utils.preProcessamento as pp
from utils.path import verificarCaminho
import os


def createDatasets(fold: int, tipo: str, diretorio: str) -> None:
    print('\nCriando datasets para o fold: {}\n'.format(fold))
    diretorioFold = os.path.join(diretorio, str(fold))
    verificarCaminho(diretorioFold)
    try:
        # Carregar o arquivo CSV
        df = pd.read_csv(os.path.join(diretorioFold, '{}{}.csv'.format(fold, tipo)))
        df['text'] = df['text'].apply(pp.remover_acentos_caracteres_especiais)
        # Selecionar apenas as colunas 'text' e 'class_label', e renomear 'class_label' para 'labels'
        df_filtered = df[['text', 'class_label']].rename(columns={'class_label': 'classe','text':'texto'})
        # Salvar o novo DataFrame em um arquivo CSV
        df_filtered.to_csv(os.path.join(diretorioFold, '{}{}_.csv'.format(fold, tipo)), index=False)
    except FileNotFoundError:
        print('Arquivo não encontrado')