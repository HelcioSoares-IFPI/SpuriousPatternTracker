import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import warnings

warnings.filterwarnings("ignore")
from utils import io
from utils import path as p
from utils import preProcessamento as pp
from utils import models as md
from utils import path as p

import pandas as pd
import numpy as np
from itertools import islice
from itertools import combinations


def filter_common_keys_by_max_value(dict1, dict2):
    """
    Filter common keys between two dictionaries by their maximum values.

    Args:
        dict1 (dict): First dictionary.
        dict2 (dict): Second dictionary.

    Returns:
        tuple: Two dictionaries with common keys filtered by their maximum values.
    """
    common_keys = set(dict1.keys()).intersection(dict2.keys())

    for key in common_keys:
        if dict1[key] > dict2[key]:
            del dict2[key]  # Remove key from dict2
        else:
            del dict1[key]  # Remove key from dict1 (includes the case of equal values)

    return dict1, dict2

def recuperarSentencasSemelhantesTreinoPorClasse(path, prefix, error_file_name):
    """
    Retrieve similar training sentences by class.

    Args:
        path (str): Base path to the files.
        prefix (str): Prefix for the file names.
        error_file_name (str): Name of the error file.

    Returns:
        list: Two dictionaries containing similar sentences for each class.
    """
    ce_uni = pd.read_csv(f'./listas/ceUnigram.csv')['uniGrama'].to_list()
    ce_bi = pd.read_csv(f'./listas/ceBigram.csv')['biGrama'].to_list()
    ces = ce_bi + ce_uni
    X_text_tr, X_train, y_train, tipo_train, ite_train, X_text_ts, X_test, y_test, class_names, tipo_test, ite_test = io.loadDatasets_ite(prefix, path)
    X_text_tr, X_train_, y_train_, tipo_train, ite_train, X_text_ts, X_test_, y_test_, class_names, tipo_test, ite_test = io.loadDatasets_ite(prefix, path)

    error_file_path = f'{path}/{error_file_name}'
    df_errors = pd.read_csv(error_file_path)
    X_e = df_errors['text_pp'].to_list()
    X_e_ = df_errors['text_pp'].to_list()
    y_e = df_errors['class_label'].to_list()
    i_e = df_errors['i'].to_list()
    tp_e = df_errors['tipo_erro'].to_list()
    X_e_ = pp.retirar_mais_frequentes(X_e, ces)
    y_e = df_errors['class_label'].to_list()

    X_train_ = pp.retirar_mais_frequentes(X_train_, ces)
    X_test_ = pp.retirar_mais_frequentes(X_test_, ces)
    semelhantes0 = dict()
    semelhantes1 = dict()

    for ii in range(len(X_e_)):
        s_ = X_e_[ii]
        # Step 2: Process the sentences with TF-IDF
        vectorizer = md.tfidfVectorizer
        tfidf_matrix = vectorizer.fit_transform(X_train_ + [s_])
        # Step 3: Calculate cosine similarity
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
        # Step 4: Find the N most similar sentences
        most_similar_indices = np.argsort(-cosine_similarities)
        most_similar_sentences = [(X_train[i], X_train_[i], cosine_similarities[i], y_train[i]) for i in most_similar_indices if y_train[i] != y_e[ii] and cosine_similarities[i] >= 0.1]

        if tp_e[ii] == 'FP':
            semelhantes0[(X_test[i_e[ii]], s_, y_e[ii])] = most_similar_sentences
        else:
            semelhantes1[(X_test[i_e[ii]], s_, y_e[ii])] = most_similar_sentences

    return [semelhantes0, semelhantes1]


def recuperaPalavrasInfluentesErro(semelhantes_por_tipo_erro):
    """
    Retrieve influential words for errors based on similar sentences.

    Args:
        semelhantes_por_tipo_erro (list): List of dictionaries containing similar sentences for each error type.

    Returns:
        list: List of dictionaries with main sentences and their similar sentences.
    """
    ms = []
    for semelhantes in semelhantes_por_tipo_erro:
        main_sentences = dict()
        for a, most_similar_sentences in semelhantes.items():
            #print(f"{a[2]} {a[1]}")
            sentence_lists = []
            for indice, sentence, similarity, y_t in most_similar_sentences:
                #print(f"SO: {indice}, Sentence: {sentence}, Similarity: {similarity}, Label: {y_t}")
                #print(f"{sentence}")
                sentence_lists.append(sentence)

            main_sentences[a[1]] = sentence_lists   
        ms.append(main_sentences)
    return ms

from itertools import permutations
import re

def ordenar_palavras_por_frequencia_aparicoes(dict_combinacoes, lista_sentencas):
    """
    Order words by frequency of appearances in sentences.

    Args:
        dict_combinacoes (dict): Dictionary of word combinations.
        lista_sentencas (list): List of sentences.

    Returns:
        dict: Dictionary with reordered word combinations by frequency.
    """
    def contar_ocorrencias(permutacao, sentencas):
        # Count how many times the permutation of words appears in the list of sentences
        pattern = r'\b' + r'.*'.join(permutacao) + r'.*\b'
        return sum(bool(re.search(pattern, sentenca, re.IGNORECASE)) for sentenca in sentencas)

    dict_ordenado = {}
    for chave, valor in dict_combinacoes.items():
        palavras = chave.split()
        # Generate all possible permutations of the words
        permutacoes = list(permutations(palavras))
        maior_contagem = 0
        permutacao_mais_frequente = chave  # Default value

        # Find the permutation with the highest count of appearances
        for permutacao in permutacoes:
            contagem_atual = contar_ocorrencias(permutacao, lista_sentencas)
            if contagem_atual > maior_contagem:
                maior_contagem = contagem_atual
                permutacao_mais_frequente = ' '.join(permutacao)
        
        # Update the dictionary with the reordered key while keeping the original value
        dict_ordenado[permutacao_mais_frequente] = valor

    return dict_ordenado

def encontrar_combinacoes01(sentenca, combinacoes):
    """
    Find combinations of words in a sentence.

    Args:
        sentenca (str): Sentence to search in.
        combinacoes (list): List of word combinations to find.

    Returns:
        list: List of found combinations.
    """
    # Normalize the sentence by removing punctuation and converting to lowercase
    sentenca_normalizada = ''.join(char.lower() for char in sentenca if char.isalnum() or char.isspace()).split()
    palavras_sentenca = set(sentenca_normalizada)
    
    # List to store the found combinations
    combinacoes_encontradas = []
    
    for combinacao in combinacoes:
        # Create a set with the words of the current combination (also normalized)
        palavras_combinacao = set(combinacao.lower().split())
        
        # Check if all words of the combination occur in the sentence
        if palavras_combinacao.issubset(palavras_sentenca):
            combinacoes_encontradas.append(combinacao)
    
    return combinacoes_encontradas

def contar_ocorrencias_combinacoes(lista_sentencas, combinacoes):
    """
    Count occurrences of word combinations in a list of sentences.

    Args:
        lista_sentencas (list): List of sentences.
        combinacoes (list): List of word combinations to count.

    Returns:
        dict: Dictionary with the count of occurrences for each combination.
    """
    # Initialize the count dictionary with zero for all combinations
    contagem_ocorrencias = {combinacao: 0 for combinacao in combinacoes}
    
    for sentenca in lista_sentencas:
        # Find the combinations that occur in the current sentence
        combinacoes_encontradas = encontrar_combinacoes01(sentenca, combinacoes)
        
        # Update the count of occurrences for the found combinations
        for combinacao in combinacoes_encontradas:
            contagem_ocorrencias[combinacao] += 1
    
    return contagem_ocorrencias

def contar_ocorrencias_semelantes(caminho, pref, md_name, sent):
    """
    Count occurrences of similar sentences and save the results.

    Args:
        caminho (str): Base path to the files.
        pref (str): Prefix for the file names.
        md_name (str): Name of the model.
        sent (str): Sentence type.

    Returns:
        list: Two dictionaries containing the count of occurrences for training and test sets.
    """
    X_train, y_train, X_test, y_test = io.loadDatasets_ite01(pref, caminho)

    words0 = list(islice(io.lerJson(f'{caminho}{p.path_if}{pref}{md_name}{io.arquivo_if_resumido}0.json').keys(), 60))
    words1 = list(islice(io.lerJson(f'{caminho}{p.path_if}{pref}{md_name}{io.arquivo_if_resumido}1.json').keys(), 60))
    fi = [words1, words0]
    semelhantes0 = f'{caminho}{p.path_if}{pref}{md_name}Semelhantes0{sent.capitalize()}.json'
    semelhantes1 = f'{caminho}{p.path_if}{pref}{md_name}Semelhantes1{sent.capitalize()}.json'
    semelhantes = [io.lerJson(semelhantes0), io.lerJson(semelhantes1)]
    
    print()
    print(sent)
    print('S0: ' + semelhantes0)
    print('S1: ' + semelhantes1)
    
    total_treino = [{}, {}]
    total_test = [{}, {}]

    for classe in range(0, 2):
        for s, semelhantes_list in semelhantes[classe].items():
            #print(f'{s=}')
            #print(f'{semelhantes_list=}')
            lista_sentencas = [sublista[0] for sublista in semelhantes_list] 
            combinacoes_encontradas = encontrar_combinacoes01(s, fi[classe])
            lista_combinacoes = combinacoes_encontradas.copy()
            lista_combinacoes.extend([' '.join(comb) for comb in combinations(combinacoes_encontradas, 2)])
            lista_combinacoes.extend([' '.join(comb) for comb in combinations(combinacoes_encontradas, 3)])
            # Count the occurrences of the combinations in the sentences
            resultado = contar_ocorrencias_combinacoes(lista_sentencas, lista_combinacoes)
            frequencia_combinacoes = contar_ocorrencias_combinacoes([s], lista_combinacoes)

            for key, value in resultado.items():
                if key in total_treino[classe]:
                    total_treino[classe][key] += value
                else:
                    total_treino[classe][key] = value

            for key, value in frequencia_combinacoes.items():
                if key in total_test[classe]:
                    total_test[classe][key] += value
                else:
                    total_test[classe][key] = value

        total_treino[classe] = {chave: valor for chave, valor in total_treino[classe].items() if valor >= 2}
        total_treino[classe] = dict(sorted(total_treino[classe].items(), key=lambda item: item[1], reverse=True))
        total_treino[classe] = ordenar_palavras_por_frequencia_aparicoes(total_treino[classe], X_train + X_test)

        total_test[classe] = dict(sorted(total_test[classe].items(), key=lambda item: item[1], reverse=True))
        total_test[classe] = ordenar_palavras_por_frequencia_aparicoes(total_test[classe], X_train + X_test)

        
        io.gravarJson(f'{caminho}{p.path_if}{pref}{md_name.upper()}ErrorTreino{classe}{sent.capitalize()}.json', dict(total_treino[classe]))
        io.gravarJson(f'{caminho}{p.path_if}{pref}{md_name.upper()}ErrorTeste{classe}{sent.capitalize()}.json', dict(total_test[classe]))

    return [total_treino, total_test]

def create_frequency_csv(test_data, training_same_data, training_other_data, csv_path):
    """
    Create a CSV file from three dictionaries containing patterns and their frequencies.
    
    Parameters:
    - test_data: Dictionary with patterns and frequencies from test data.
    - training_same_data: Dictionary with patterns and frequencies from training data (same).
    - training_other_data: Dictionary with patterns and frequencies from training data (other).
    - csv_path: Path to save the CSV file.
    
    Returns:
    - csv_path: The path where the CSV file is saved.
    """

    # Extract the common keys (patterns)
    common_patterns = set(test_data.keys()) & set(training_same_data.keys()) & set(training_other_data.keys())

    # Create a DataFrame
    df = pd.DataFrame([
        {
            "padrao": pattern,
            "freqTreinoMesmaClasse": training_other_data[pattern],
            "freqTreinoOutraClasse": training_same_data[pattern],
            "freqErroTesteOutraClasse": test_data[pattern]
        }
        for pattern in common_patterns
    ])

    # Calculate the sum and sort
    df['total'] = df[['freqTreinoMesmaClasse', 'freqTreinoOutraClasse', 'freqErroTesteOutraClasse']].sum(axis=1)
    df_sorted = df.sort_values(by='total', ascending=False).drop(columns=['total'])

    # Save to CSV
    df_sorted.to_csv(csv_path, index=False)
    
    return csv_path

def contar_ocorrencias_semelantes_teste_mesmo(caminho, pref, md_name):
    """
    Count occurrences of similar sentences in the same test set and save the results.

    Args:
        caminho (str): Base path to the files.
        pref (str): Prefix for the file names.
        md_name (str): Name of the model.

    Returns:
        None
    """
    sent_erros = [list(io.lerJson(f'{caminho}{p.path_if}{pref}{md_name}Semelhantes0Mesmo.json')), list(io.lerJson(f'{caminho}{p.path_if}{pref}{md_name}Semelhantes1Mesmo.json'))]
    comb_erros = [list(io.lerJson(f'{caminho}{p.path_if}{pref}{md_name}ErrorTeste1Outro.json')), list(io.lerJson(f'{caminho}{p.path_if}{pref}{md_name}ErrorTeste0Outro.json'))]
    error_total = [{}, {}]

    for classe in range(0, 2):
        sentencas = sent_erros[classe]
        for comb in comb_erros[classe]:
            # Words to be searched
            palavras = comb.split(' ')

            # Creating the regular expression
            if len(palavras) == 1:
                regex = r'(?=.*\b{}\b)'.format(*palavras)
            elif len(palavras) == 2:
                regex = r'(?=.*\b{}\b)(?=.*\b{}\b)'.format(*palavras)
            else:
                regex = r'(?=.*\b{}\b)(?=.*\b{}\b)(?=.*\b{}\b)'.format(*palavras)

            # Compiling the regular expression for better performance
            compiled_regex = re.compile(regex, re.IGNORECASE)

            # Counting the occurrences
            ocorrencias = sum(bool(compiled_regex.search(sentenca)) for sentenca in sentencas)
            
            if comb in error_total[classe].keys():
                error_total[classe][comb] += ocorrencias
            else:
                error_total[classe][comb] = ocorrencias
        
        io.gravarJson(f'{caminho}{p.path_if}{pref}{md_name.upper()}ErrorTeste{1-classe}Mesmo.json', dict(error_total[classe]))
                    
        df = pd.DataFrame(list(error_total[classe].items()), columns=['padrao', 'freqErroTesteMesmaClasse'])
        # Check if the column 'freqErroTesteMesmaClasse' exists in the DataFrame

        dfFE = pd.read_csv(f'{caminho}{p.path_if}{pref}{md_name}ErrorFeatures{classe}.csv')
        if 'freqErroTesteMesmaClasse' in dfFE.columns:
            # If it exists, delete it
            dfFE.drop(columns=['freqErroTesteMesmaClasse'], inplace=True)

        merged_df = pd.merge(dfFE, df, on='padrao', how='inner')
        # Reorder the columns as desired
        nova_ordem_colunas = ['padrao', 'freqTreinoMesmaClasse', 'freqTreinoOutraClasse', 'freqErroTesteMesmaClasse', 'freqErroTesteOutraClasse']
        merged_df = merged_df.reindex(columns=nova_ordem_colunas)
        merged_df.to_csv(f'{caminho}{p.path_if}{pref}{md_name}ErrorFeatures{classe}.csv', index=False)
