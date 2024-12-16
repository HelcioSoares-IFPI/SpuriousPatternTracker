from tqdm.auto import tqdm
import os
import pandas as pd
import numpy as np
import json
from operator import itemgetter
import shutil
from itertools import islice
from sklearn.preprocessing import PowerTransformer


def gravarJson(nomeArquivo,obj):
    with open(nomeArquivo, 'w', encoding='UTF-8') as f:
        str = json.dumps(obj)
        f.write(str)


def lerJson(nomeArquivo):
    with open(nomeArquivo, encoding='UTF-8') as f:
        s = f.read()
        data = json.loads(s)        
    return data


# Juntar Dados de Diferentes K-Folds
def juntarDados(clf: str, caminho: str, classe=None, tipo = None):
    dados = []
    for kfold in tqdm(range(1, 6)):
        if classe is None:
            # Tratamento para CE
            arquivos = [os.path.join(caminho, str(kfold), 'if', '{}bertCEnn.csv'.format(kfold))]
            tipo_arquivo = 'csv'
        else:
            # Tratamento para IF e EF
            if tipo == 'if':
                arquivos = [os.path.join(caminho, str(kfold), 'if', '{}0bert{}.json'.format(kfold, classe))]
                tipo_arquivo = 'json'
            else: # tipo == 'ef'
                arquivos = [os.path.join(caminho, str(kfold), 'if', '{}bertErrorFeatures{}.csv'.format(kfold, classe))]
                tipo_arquivo = 'csv'
        
        for arquivo in arquivos:
            if os.path.exists(arquivo):
                if tipo_arquivo == 'csv':
                    df = pd.read_csv(arquivo)
                    dados.append(df)
                elif tipo_arquivo == 'json':
                    with open(arquivo, 'r') as f:
                        lista = json.load(f)
                        nova_lista = []
                        for key, value in lista.items():
                            nova_lista.append([key, value])
                        dados.extend(nova_lista)
    pref_validacao = '6'
    caminho_validacao = os.path.join(caminho, pref_validacao, 'if')
    if not os.path.exists(caminho_validacao):
        os.makedirs(caminho_validacao)

    if classe is None:
        arquivo_final = os.path.join(caminho_validacao, '{}0bertCEnn.csv'.format(pref_validacao))
        dados_concatenados = pd.concat(dados)
        dados_concatenados = dados_concatenados.sort_values(by='padrao', ascending = True)
        dados_concatenados_false = dados_concatenados[dados_concatenados['cond'] == False]
        dados_concatenados = dados_concatenados[dados_concatenados['cond'] == True]
        dados_concatenados.to_csv(arquivo_final, index=False)

        position = len(arquivo_final) - 4
        arquivo_final_false = arquivo_final[:position] + '_false' + arquivo_final[position:]
        dados_concatenados_false.to_csv(arquivo_final_false, index=False)
    else:
        if tipo == 'if':
            arquivo_final = os.path.join(caminho_validacao, '{}0bertListClass{}.json'.format(pref_validacao, classe))
            gravarJson(arquivo_final, dados)
        else: # tipo == 'ef'
            arquivo_final = os.path.join(caminho_validacao, '{}0bertErrorFeatures{}.csv'.format(pref_validacao, classe))
            dados_concatenados = pd.concat(dados)
            dados_concatenados.to_csv(arquivo_final, index=False)


# Sumarizar Correlações Espúrias
def sumarizar_ce(caminho, pref, md_name):

    def find_cl_value(palavra):
        # Encontrando o valor de 'cl' para a primeira ocorrência da 'palavra' no dataframe true_df
        cl_value = cond_true_df[cond_true_df['padrao'] == palavra]['cl'].iloc[0]
        return cl_value
    nome_arquivo = os.path.join(caminho, pref, 'if', '{}0bertCEnn.csv'.format(pref))
    nome_arquivo_final = os.path.join(caminho, pref, 'if', '{}0bertCE'.format(pref))
    # Lendo o arquivo CSV
    df = pd.read_csv(nome_arquivo)

    # Passo 2: Dividir o dataframe em dois com base no valor da coluna 'cond' e criar os arquivos CSV
    cond_true_df  = df[df['cond'] == True]
    cond_false_df = df[df['cond'] == False]

    
    # Perform the correct aggregation
    aggregated_df_only_palavras = df.groupby('padrao', as_index=False).agg(
        cl=('cl', 'first'),
        padrao=('padrao', 'first'),
        cat =('cat','first'),
        pIdx=('p.idx', 'sum'),
        pQtd=('p.qtd', 'sum'),
        aIdx=('a.idx', 'sum'),
        aQtd=('a.qtd', 'sum'),
    )
    
    # Primeiro, vamos obter um dataframe com 'palavras', 'iceTotal', e 'qtdTotal' ordenado por 'qtdTotal'
    final_df_ordered = aggregated_df_only_palavras
    
    # Aplicando a função para criar a coluna 'cl' no dataframe final_df_ordered
    final_df_ordered['cl'] = final_df_ordered['padrao'].apply(find_cl_value)

    final_df_ordered.to_csv(os.path.join(caminho, pref, 'if', '{}0bertTemp00.csv'.format(pref)))

    nova_ordem = ['cl','padrao','cat','pIdx','pQtd','aIdx','aQtd']
    final_df_ordered = final_df_ordered[nova_ordem]

    final_df_0 = final_df_ordered[final_df_ordered['cl'] == 0]      
    final_df_1 = final_df_ordered[final_df_ordered['cl'] == 1]     

    final_df_0.to_csv(f'{nome_arquivo_final}0.csv', index=False)
    final_df_1.to_csv(f'{nome_arquivo_final}1.csv', index=False)


def createWords(features):
    words = []
    for tupla in features:
        word = tupla[0]
        if word not in words:
            words.append(word)
    return sorted(words)


def createEmptyVocab(features):
    vocab = dict()
    for tupla in features:
        word = tupla[0]
        vocab[word] = []
    return vocab
  

def createValues(features):
    vocab = createEmptyVocab(features)
    for tupla in features:
        #for tupla in feature:
        word  = tupla[0]
        value = tupla[1]
        vocab[word].append(value)
    #record = [0]*len(vocab)
    return vocab


def calcDictSqrt(features):
    words = createWords(features)
    vocab = createValues(features)
    dicSqr = dict()
    for key in vocab:
        vals = vocab[key]
        vals_np = np.array(vals)
        sum_ = np.sum(vals_np)
        sqr = np.sqrt(sum_)
        dicSqr[key] = sqr
    return dicSqr


def normalize_dictionary_values(dictionary):
    # Obtém o valor máximo e mínimo dos valores do dicionário
    values = list(dictionary.values())
    max_value = max(values)
    min_value = min(values)

    # Normaliza os valores
    normalized_dict = {}
    for key, value in dictionary.items():
        max_value_min_value = 000000000.1 if (max_value - min_value) == 0 else max_value - min_value
        normalized_value = (value - min_value) / (max_value_min_value)
        normalized_dict[key] = normalized_value

    return normalized_dict


#Carregar arquivos do Lime    
def loadLime(caminho, pref, md_name):
    mapFeaturesModel = dict()
    caminho_if = os.path.join(caminho, pref, 'if')
    mapMeans = dict()
    md_name
    file_name0 = os.path.join(caminho_if, '{}0{}ListClass0.json'.format(pref, md_name))
    if os.path.exists(file_name0):
        features = lerJson(file_name0)
        mapMeans[0] = calcDictSqrt(features)
        mapMeans[0] = normalize_dictionary_values(mapMeans[0])
    else:
        print(f'{file_name0} Não existe')
    file_name1 = os.path.join(caminho_if, '{}0{}ListClass1.json'.format(pref, md_name))
    if os.path.exists(file_name1):
        features = lerJson(file_name1)
        mapMeans[1] = calcDictSqrt(features)
        mapMeans[1] = normalize_dictionary_values(mapMeans[1])
    mapFeaturesModel[md_name] =  mapMeans    
    return mapFeaturesModel


def loadImportanteFeatures(caminho, pref, md_name):
    mapFeaturesLime  = loadLime(caminho, pref, md_name)
    return mapFeaturesLime


def lerIf(caminho, pref, md_name, explainer):   
    #Ler as featrues importantes
    mapFeaturesLime  = dict()
    mapFeaturesLime = loadImportanteFeatures(caminho, pref, md_name)
    mapFeatures = mapFeaturesLime
        
    #Ler as features do classificador específico
    f0 = mapFeatures[md_name][0]
    f0 = dict(sorted(f0.items(), key = itemgetter(1), reverse = False))
    f0 = {key: value for key, value in f0.items()}
    
    
    f1 = mapFeatures[md_name][1]
    f1 = dict(sorted(f1.items(), key = itemgetter(1), reverse = False))
    f1 = {key: value for key, value in f1.items()}
    
    f_global = [f0,f1]
    
    f0o = dict(sorted(f0.items(), key = itemgetter(1), reverse = True))
    f0o = {key: value for key, value in f0o.items()}
    
    f1o = dict(sorted(f1.items(), key = itemgetter(1), reverse = True))
    f1o = {key: value for key, value in f1o.items()}

    gravarJson(os.path.join(caminho, pref, 'if', '{}0{}IF0.json'.format(pref, md_name)), f0o)
    gravarJson(os.path.join(caminho, pref, 'if', '{}0{}IF1.json'.format(pref, md_name)), f1o)    
    return f_global


def filter_common_keys_by_max_value(dict1, dict2):
    common_keys = set(dict1.keys()).intersection(dict2.keys())

    for key in common_keys:
        if dict1[key] > dict2[key]:
            del dict2[key]  # Remover chave de dict2
        else:
            del dict1[key]  # Remover chave de dict1 (inclui o caso de valores iguais)

    return dict1, dict2


# Sumarizar Importância de Features
def sumarizar_if(caminho, pref, md_name):
    f_global = lerIf(caminho, pref, md_name,'Lime')    
    f_global[0]= dict(sorted(f_global[0].items(), key = itemgetter(1), reverse = True))
    f_global[1]= dict(sorted(f_global[1].items(), key = itemgetter(1), reverse = True))
    f_global[0], f_global[1] = filter_common_keys_by_max_value(f_global[0], f_global[1])
    gravarJson(os.path.join(caminho, pref, 'if', '{}0{}IF0.json'.format(pref, md_name)), f_global[0])
    gravarJson(os.path.join(caminho, pref, 'if', '{}0{}IF1.json'.format(pref, md_name)), f_global[1])


# Sumarizar Erros de Features
def sumarizar_ef(caminho, pref, md_name):
    def sumarize_er(caminho, pref, md_name,classe):
        # Carregar os dados do arquivo CSV
        df = pd.read_csv(os.path.join(caminho, pref, 'if', '{}0{}ErrorFeatures{}.csv'.format(pref, md_name, classe)))
        # Agrupar os dados pela primeira coluna (assumindo que é uma coluna de identificação ou categoria)
        # e somar as colunas 'frequenciaTreino' e 'frequenciaErrosTeste'
        df_agrupado = df.groupby('padrao', as_index=False).agg({
            'freqTreinoMesmaClasse': 'sum',
            'freqTreinoOutraClasse': 'sum',
            'freqErroTesteMesmaClasse': 'sum',
            'freqErroTesteOutraClasse': 'sum'
        })

        # Adiciona uma nova coluna 'soma_frequencias' que é a soma de 'frequenciaTreino' e 'frequenciaErrosTeste'
        df_agrupado['div_frequenciasOM'] = round(df_agrupado['freqTreinoOutraClasse']    / df_agrupado['freqTreinoMesmaClasse'],3)
        df_agrupado['div_freqErroOM']    = round(df_agrupado['freqErroTesteOutraClasse'] / df_agrupado['freqErroTesteMesmaClasse'] ,3)

        # Aplicar o filtro: soma de 'frequenciaTreino' + 'frequenciaErrosTeste' > 100
        df_filtrado = df_agrupado[df_agrupado['freqErroTesteMesmaClasse'] >= 3]
        #df_filtrado = df_filtrado.sort_values(by='div_frequenciasMO', ascending=False)


        # Salvar o resultado em outro arquivo CSV
        caminho_arquivo_saida = os.path.join(caminho, pref, 'if', '{}0{}EF{}.csv'.format(pref, md_name, classe))
        df_filtrado.to_csv(caminho_arquivo_saida, index=False)

    sumarize_er(caminho, pref, md_name, 0)
    sumarize_er(caminho, pref, md_name, 1)


# Mover Arquivos auxiliares para os calculos
def mover_arquivos(caminho, pref, md_name):
    # Caminho da pasta onde os arquivos estão localizados
    caminho_pasta = os.path.join(caminho, pref, 'if')
    
    # Caminho da subpasta para onde os arquivos devem ser movidos
    caminho_subpasta = os.path.join(caminho_pasta, 'todos')
    pa = f'{pref}{0}{md_name}'
    # Lista dos nomes dos arquivos que não devem ser movidos
    arquivos_excluir = [  f'{pa}CEnn.csv' , f'{pa}ListClass0.json', f'{pa}ListClass1.json'
                        , f'{pa}ErrorFeatures0.csv', f'{pa}ErrorFeatures1.csv']
    # Criar a subpasta se ela não existir
    if not os.path.exists(caminho_subpasta):
        os.makedirs(caminho_subpasta)
    
    # Percorrer todos os arquivos na pasta
    for arquivo in os.listdir(caminho_pasta):
        # Verificar se o arquivo atual não está na lista de exclusão e é um arquivo (não uma pasta)
        if arquivo in arquivos_excluir and os.path.isfile(os.path.join(caminho_pasta, arquivo)):
            # Caminho completo do arquivo atual
            caminho_completo_arquivo = os.path.join(caminho_pasta, arquivo)
            
            # Caminho completo do destino do arquivo na subpasta
            destino_arquivo = os.path.join(caminho_subpasta, arquivo)
            
            # Mover o arquivo para a subpasta
            shutil.move(caminho_completo_arquivo, destino_arquivo)


# Função para preparar o arquivo csv que será calculada a importancia final de cada padrão.
def calcular_peso_global(frase, dicionario):
    palavras = frase.split()  # Divide a frase em palavras
    pesos = [dicionario.get(palavra, 0) for palavra in palavras]  # Lista de pesos para cada palavra
    
    if len(pesos) == 0:
        return 0  # Retorna 0 se a lista de pesos estiver vazia para evitar divisão por zero
    else:
        media = sum(pesos) / len(pesos)  # Calcula a média dos pesos
        return round(media, 4)
    

#Aplicar no dataset 6 e, opcionalmente, nos demais datasets.
def criarImportanciaErros(caminho, pref, md_name, class_index):

#Nome dos arquivos
#arquivo_if = 'ListClass'
#arquivo_er = 'ErrorFeatures'
#arquivo_ce = 'CEnn'
#
#arquivo_ce_resumido = 'CE'
#arquivo_if_resumido = 'IF'
#arquivo_er_resumido = 'EF'

    df_freq_erros = pd.read_csv(os.path.join(caminho, pref, 'if', '{}0{}EF{}.csv'.format(pref, md_name, class_index))) #EF
    df_ce = pd.read_csv(os.path.join(caminho, pref, 'if', '{}0{}CE{}.csv'.format(pref, md_name, 1-class_index))) #CE
    global_weights = lerJson(os.path.join(caminho, pref, 'if', '{}0{}IF{}.json'.format(pref, md_name, class_index)))  #IF
    global_weights = dict(islice(global_weights.items(), 90))

    #Limita as features importance que são iguais às errors features
    df_filtrado = df_freq_erros[df_freq_erros['padrao'].apply(lambda x: all(palavra in global_weights for palavra in x.split()))]

    df_filtrado['peso_global'] = df_filtrado['padrao'].apply(lambda x: calcular_peso_global(x, global_weights))

    # Realizar a junção dos DataFrames baseada na coluna 'padrao'
    df_final = pd.merge(df_filtrado, df_ce, on='padrao')
    df_final = df_final.drop(columns=['cl'])
    df_final['div_frequenciasOM'] = round(100*df_final['div_frequenciasOM'],3)
    df_final['div_freqErroOM']    = round(100*df_final['div_freqErroOM'],3)
    df_final['peso_global']       = round(100*df_final['peso_global'],3)
    df_final['pIdx']              = round(df_final['pIdx'],3)

    #df_final['qtdTotal'] = scaler01.fit_transform(np.array(df_final['qtdTotal']).reshape(-1, 1))
    df_final.to_csv(os.path.join(caminho, pref, 'if', '{}0{}IfErrorsInfluence{}.csv'.format(pref, md_name, class_index)), index=False)
    return

def semNormSemPeso(caminho, pref, md_name):
    colunas01 = ['freqTreinoMesmaClasse','freqTreinoOutraClasse','div_frequenciasOM','peso_global','iceTotal','qtdTotal']
    colunas01 = ['freqTreinoMesmaClasse','freqTreinoOutraClasse','freqErroTesteMesmaClasse','freqErroTesteOutraClasse','div_frequenciasOM','div_freqErroOM','peso_global','iceTotal','qtdTotal']
    scaler03 = PowerTransformer ('yeo-johnson')
    for classe in range(2):
        #Metodos com normalização, com e sem pesos
        df_SemNormSemPeso = pd.read_csv(os.path.join(caminho, pref, 'if', '{}0{}IfErrorsInfluence{}.csv'.format(pref, md_name, classe)))
        df_SemNormSemPeso['importancia'] = df_SemNormSemPeso.apply(lambda row: (row[colunas01].sum()) / len(colunas01), axis=1)
        df_SemNormSemPeso['importancia'] = scaler03.fit_transform(np.array(df_SemNormSemPeso['importancia']).reshape(-1, 1))
        # Ordenar e salvar o resultado
        df_SemNormSemPeso = df_SemNormSemPeso.sort_values(by='importancia', ascending=False)
        arquivo_saida = f'{caminho}{p.path_if}{pref}{md_name}R{classe}SemNormSempeso.csv'
        df_SemNormSemPeso.to_csv(arquivo_saida, index=False)

    df0 = pd.read_csv(f'{caminho}{p.path_if}{pref}{md_name}R0SemNormSemPeso.csv')
    df1 = pd.read_csv(f'{caminho}{p.path_if}{pref}{md_name}R1SemNormSemPeso.csv')
    titulo_fig = f'{md_name}RSemNormSemPeso'
    #pf.plotCE(df0, df1, n, f'{caminho}{pref}{md_name}RSemNormSemPeso.png',titulo_fig) 


def calcular_importancia(caminho, pref, md_name):
    semNormSemPeso(caminho, pref, md_name)
    semNormComPeso(caminho, pref, md_name)
    comNormSemPeso(caminho, pref, md_name)
    comNormComPeso(caminho, pref, md_name)


def gerar_resultados(caminho: str):
    md_name = 'bert'
    juntarDados(md_name, caminho)
    juntarDados(md_name, caminho, 0, 'if')
    juntarDados(md_name, caminho, 1, 'if')
    juntarDados(md_name, caminho, 0, 'ef')
    juntarDados(md_name, caminho, 1, 'ef')
    print('Etapa 10: Sumarizar dados', end="")

    pref = '6'
    sumarizar_ce(caminho, pref, md_name)
    sumarizar_if(caminho, pref, md_name)
    sumarizar_ef(caminho, pref, md_name)
    mover_arquivos(caminho, pref, md_name)
#
    print('Etapa 11: Calcular resultado', end="")

    criarImportanciaErros(caminho, pref, md_name, 0)
    criarImportanciaErros(caminho, pref, md_name, 1)
    #calcular_importancia(caminho,   pref, md_name)
    #criarArquivoComparacao(caminho, pref, md_name, '0')
    #criarArquivoComparacao(caminho, pref, md_name, '1')

#gerar_resultados('/var/projetos/Jupyterhubstorage/victor.silva/CorrelacoesEspurias/dts_completo')