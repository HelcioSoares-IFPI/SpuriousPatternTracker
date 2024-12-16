import os
from utils.utils import load_dict_from_json
from tqdm.auto import tqdm
import pandas as pd
from utils import io
from utils import path as p
from itertools import islice


def calc_cat(cl, dif_esp, dif_sen):
    if dif_esp < 0 and dif_sen < 0:
        return 0 #Palavra importante para as duas classes

    if cl == 0:
        if   dif_esp < 0 and dif_sen > 0:
            return 4   #Classe 0 palavras 1, diminui a especificidade e aumenta a sensibilidade. Não é correlação espúria.
        elif dif_esp > 0 and dif_sen < 0:
            return 1    #Classe 0 palavras 1, aumenta a especificidade e diminui a sensibilidade. Pode ser correlacao espúria.
        elif dif_esp > 0 and dif_sen == 0:
            return 2  #Classe 0 palavras 1, aumenta a especificidade e não altera a sensibilidade. Pode ser correlacao espúria.
    elif cl == 1:
        if   dif_sen < 0 and dif_esp > 0:
            return 5   #Classe 1 palavras 0, aumenta a especificidade e diminue a sensibilidade. Não é correlação espúria.
        elif dif_sen > 0 and dif_esp < 0:
            return  1   #Classe 1 palavras 0, aumenta a sensibilidade e diminui a especificidade. Pode ser correlacao espúria.
        elif dif_sen > 0 and dif_esp == 0:
            return  2 #Classe 1 palavras 0, aumenta a sensibilidade e não altera a especificidade. Pode ser correlacao espúria.

    if dif_esp > 0 and dif_sen > 0:
        return 3 #Aumenta as duas métricas
     
    return 4 #Sobras


def formatar_saida(mf, mi):
    # Define as métricas a serem comparadas
    metricas = ['eval_acc', 'eval_pre', 'eval_f1', 'eval_esp', 'eval_tn', 'eval_fn', 'eval_sen', 'eval_tp', 'eval_fp']
    saida = []
    # Itera sobre cada métrica, obtendo os valores de mf e mi, e calcula a diferença
    for metrica in metricas:   
        valor_mi  = mi[metrica]
        valor_mf  = mf[metrica]
        diferenca = abs(valor_mf - valor_mi)

        # Adiciona os valores na lista de saída
        saida.extend([valor_mi, valor_mf, round(diferenca,3)])

    # Converte a lista em uma string formatada
    saida_formatada = ','.join(map(str, saida))
    return saida_formatada


def somar_acertos(a, b):
    resultado = a + b
    # Retorna o resultado arredondado para três casas decimais
    return round(resultado, 3)


def somar_perturbacao(a, b):
    # Verifica se ambos os números são negativos
    if a < 0 and b < 0:
        # Se ambos são negativos, soma os valores absolutos e torna o resultado negativo
        resultado = -(abs(a) + abs(b))
    else:
        # Para outros casos (pelo menos um número positivo ou zero), soma os valores absolutos
        resultado = abs(a) + abs(b)
    return round(resultado, 3)


def printCorrelacao(padrao, cl, pref, tipo, mf, mi):
    cor = []
    #print(metricasFinais['eval_esp'])
    mf_esp, mf_sen, mf_tp, mf_tn, mf_fn, mf_fp = mf['eval_esp'], mf['eval_sen'], mf['eval_tp'], mf['eval_tn'], mf['eval_fn'], mf['eval_fp']
    mi_esp, mi_sen, mi_tp, mi_tn, mi_fn, mi_fp = mi['eval_esp'], mi['eval_sen'], mi['eval_tp'], mi['eval_tn'], mi['eval_fn'], mi['eval_fp']

    dif_esp = round(mf_esp - mi_esp, 3)
    dif_sen = round(mf_sen - mi_sen, 3)
    dif_tp  = round(mf_tp  - mi_tp,  3)
    dif_tn  = round(mf_tn  - mi_tn,  3)
    dif_fn  = round(mf_fn  - mi_fn,  3)
    dif_fp  = round(mf_fp  - mi_fp,  3)

    corEspuria = dif_sen > 0 or dif_esp > 0

    pIdx  = somar_perturbacao(dif_sen,dif_esp) #Índice de perturbação
    pQtd  = somar_perturbacao(dif_fp, dif_fn)  #Quantidade de itens perturbados (Que mudam de previsão após a perturbação da base)
    aIdx  = somar_acertos(dif_esp ,dif_sen) #Índice de acerto 
    aQtd  = somar_acertos(dif_tp, dif_tn) #Quantidade de acertos após a perturbação

    saida_formatada = formatar_saida(mf, mi)
    cat = calc_cat(cl, dif_esp, dif_sen)
    cor.append(f'{tipo},{pref},{cl},{saida_formatada},{pIdx},{pQtd},{aIdx},{aQtd},{cat},{corEspuria},{padrao}')
    return cor


def lista_para_csv(lista, nome_arquivo_csv):
    # A primeira linha da lista contém os cabeçalhos, então a separamos dos dados
    cabeçalhos = lista[0].split(',')
    dados = [linha.split(',') for linha in lista[1:]]
    dataframe = pd.DataFrame(dados, columns=cabeçalhos)
    dataframe.to_csv(nome_arquivo_csv, index=False)

# Função para preparar o arquivo csv que será calculada a importancia final de cada padrão.
def calcular_peso_global(frase, dicionario):
    palavras = frase.split()  # Divide a frase em palavras
    pesos = [dicionario.get(palavra, 0) for palavra in palavras]  # Lista de pesos para cada palavra
    
    if len(pesos) == 0:
        return 0  # Retorna 0 se a lista de pesos estiver vazia para evitar divisão por zero
    else:
        media = sum(pesos) / len(pesos)  # Calcula a média dos pesos
        return round(media, 4)


def calculaPadroes(diretorio: str, fold: str, tipo='Teste') -> None:
    diretorioIF = os.path.join(diretorio, str(fold), 'if')
    words1, words0, metricasIniciais = load_dict_from_json(os.path.join(diretorioIF, '{}bertPerturbacao.json'.format(str(fold))))
    maior_tamanho = max(len(words0), len(words1))
    cabecalho = ['tp,pre,cl,i.acc,f.acc,d.acc,i.pre,f.pre,d.pre,i.f1,f.f1,d.f1,i.esp,f.esp,d.esp,i.tn,f.tn,d.tn,i.fn,f.fn,d.fn,i.sen,f.sen,d.sen,i.tp,f.tp,d.tp,i.fp,f.fp,d.fp,p.idx,p.qtd,a.idx,a.qtd,cat,cond,padrao']
    ces0, ces1 = ([] for _ in range(2))

    for i in tqdm(range(maior_tamanho), desc='Calculando padrões para o fold {}..'.format(fold), colour='yellow'):
        if i < len(words0):
            saida0 = list(words0)[i]
            metricasFinais = words0[saida0]
            ces0.extend(printCorrelacao(saida0, 0, fold, tipo, metricasFinais, metricasIniciais))
        
        if i < len(words1):
            saida1 = list(words1)[i]
            metricasFinais = words1[saida1]
            ces1.extend(printCorrelacao(saida1, 1, fold, tipo, metricasFinais, metricasIniciais))
    
    cabecalho.extend(ces0)
    cabecalho.extend(ces1)

    nome_arquivo = os.path.join(diretorioIF, '{}bert{}.csv'.format(str(fold), io.arquivo_ce))
    lista_para_csv(cabecalho, nome_arquivo)