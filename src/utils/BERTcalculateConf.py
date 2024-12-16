import warnings
warnings.filterwarnings("ignore")
from tqdm.auto import tqdm
import pandas as pd
from utils import io
from utils import path as p
import os
from itertools import islice, combinations, permutations
import re

from transformers import BertTokenizer, BertModel
import torch
import faiss
import numpy as np

def create_frequency_csv(test_data, training_same_data, training_other_data, csv_path):
    """
    Create a CSV file from three dictionaries containing patterns and their frequencies.

    Parameters:
    - test_data: Dictionary with patterns and frequencies from test data.
    - training_same_data: Dictionary with patterns and frequencies from training data (same class).
    - training_other_data: Dictionary with patterns and frequencies from training data (opposite class).
    - csv_path: Path to save the CSV file.

    Returns:
    - csv_path: The path where the CSV file is saved.
    """
    print("Creating CSV file...")

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

    # Save to CSV
    df.to_csv(csv_path, index=False)

    return csv_path

def contar_ocorrencias_semelantes_teste_mesmo(diretorio: str, fold: str, md_name: str):
    """
    Count occurrences of word combinations in similar sentences (same class) from test errors.

    Parameters:
    - diretorio: Directory containing the data.
    - fold: Fold identifier for cross-validation.
    - md_name: Model name (e.g., BERT).
    """
    sent_erros = [
        list(io.lerJson(os.path.join(diretorio, fold, p.p_inf, f'{fold}{md_name}Semelhantes0Mesmo.json'))),
        list(io.lerJson(os.path.join(diretorio, fold, p.p_inf, f'{fold}{md_name}Semelhantes1Mesmo.json')))
    ]
    comb_erros = [
        list(io.lerJson(os.path.join(diretorio, fold, p.p_inf, f'{fold}{md_name.upper()}ErrorTeste1Outro.json'))),
        list(io.lerJson(os.path.join(diretorio, fold, p.p_inf, f'{fold}{md_name.upper()}ErrorTeste0Outro.json')))
    ]
    error_total = [{}, {}]

    for classe in range(2):
        print(f"Counting word combinations for class {classe}...")
        sentencas = sent_erros[classe]
        for comb in tqdm(comb_erros[classe]):
            # Split combination into words
            palavras = comb.split(' ')

            # Create regular expression based on the number of words
            regex = r'(?=.*\b{})' + r'(?=.*\b{})' * (len(palavras) - 1)
            regex = regex.format(*palavras)

            # Compile the regex for better performance
            compiled_regex = re.compile(regex, re.IGNORECASE)

            # Count occurrences
            ocorrencias = sum(bool(compiled_regex.search(sentenca)) for sentenca in sentencas)

            if comb in error_total[classe]:
                error_total[classe][comb] += ocorrencias
            else:
                error_total[classe][comb] = ocorrencias

        # Save results as JSON and merge with existing CSV
        io.gravarJson(os.path.join(diretorio, fold, p.p_inf, f'{fold}{md_name.upper()}ErrorTeste{1 - classe}Mesmo.json'), dict(error_total[classe]))

        df = pd.DataFrame(list(error_total[classe].items()), columns=['padrao', 'freqErroTesteMesmaClasse'])
        dfFE = pd.read_csv(os.path.join(diretorio, fold, p.p_inf, f'{fold}{md_name}ErrorFeatures{classe}.csv'))

        if 'freqErroTesteMesmaClasse' in dfFE.columns:
            dfFE.drop(columns=['freqErroTesteMesmaClasse'], inplace=True)

        merged_df = pd.merge(dfFE, df, on='padrao', how='inner')
        nova_ordem_colunas = ['padrao', 'freqTreinoMesmaClasse', 'freqTreinoOutraClasse', 'freqErroTesteMesmaClasse', 'freqErroTesteOutraClasse']
        merged_df = merged_df.reindex(columns=nova_ordem_colunas)

        print(f"Saving CSV file for class {classe}...")
        merged_df.to_csv(os.path.join(diretorio, fold, p.p_inf, f'{fold}{md_name}ErrorFeatures{classe}.csv'), index=False)




def encontrar_combinacoes(sentenca, combinacoes):
    """
    Find combinations of words that appear within a given sentence.

    Parameters:
    - sentenca: Sentence to evaluate.
    - combinacoes: List of combinations to match against.

    Returns:
    - combinacoes_encontradas: List of combinations found in the sentence.
    """
    # Normalize the sentence by removing punctuation and converting to lowercase
    sentenca_normalizada = ''.join(char.lower() for char in sentenca if char.isalnum() or char.isspace()).split()
    palavras_sentenca = set(sentenca_normalizada)

    # List to store found combinations
    combinacoes_encontradas = []

    for combinacao in combinacoes:
        # Create a set with the words of the current combination (also normalized)
        palavras_combinacao = set(combinacao.lower().split())

        # Check if all words in the combination occur in the sentence
        if palavras_combinacao.issubset(palavras_sentenca):
            combinacoes_encontradas.append(combinacao)

    return combinacoes_encontradas

def contar_ocorrencias_combinacoes(lista_sentencas, combinacoes):
    """
    Count occurrences of specific combinations within a list of sentences.

    Parameters:
    - lista_sentencas: List of sentences to search in.
    - combinacoes: List of combinations to count.

    Returns:
    - contagem_ocorrencias: Dictionary with combinations as keys and their count as values.
    """
    # Initialize the count dictionary with zero for all combinations
    contagem_ocorrencias = {combinacao: 0 for combinacao in combinacoes}

    for sentenca in lista_sentencas:
        # Find combinations that occur in the current sentence
        combinacoes_encontradas = encontrar_combinacoes(sentenca, combinacoes)

        # Update the occurrence count for the found combinations
        for combinacao in combinacoes_encontradas:
            contagem_ocorrencias[combinacao] += 1

    return contagem_ocorrencias

def contar_ocorrencias_semelantesErro(diretorio: str, fold: int, md_name: str, sent: str) -> list:
    """
    Count occurrences of word combinations for sentences similar to the error samples.

    Parameters:
    - diretorio: Directory containing the data.
    - fold: Fold identifier for cross-validation.
    - md_name: Model name (e.g., BERT).
    - sent: Type of sentence ("outro" or "mesmo").

    Returns:
    - List containing two dictionaries:
        - total_treino: Frequencies of word combinations in the training data.
        - total_test: Frequencies of word combinations in the test data.
    """
    # Load training and test data
    X_train, _, X_test, _ = io.loadDatasets_ite01(fold, os.path.join(diretorio, fold) + '/')
    
    # Load important words for each class
    words0 = list(islice(io.lerJson(os.path.join(diretorio, fold, p.p_inf, f'{fold}0{md_name}0.json')).keys(), 60))
    words1 = list(islice(io.lerJson(os.path.join(diretorio, fold, p.p_inf, f'{fold}0{md_name}1.json')).keys(), 60))
    fi = [words1, words0]

    # Load similar sentences
    semelhantes = [
        io.lerJson(os.path.join(diretorio, fold, p.p_inf, f'{fold}{md_name}Semelhantes0{sent.capitalize()}.json')),
        io.lerJson(os.path.join(diretorio, fold, p.p_inf, f'{fold}{md_name}Semelhantes1{sent.capitalize()}.json'))
    ]

    total_treino = [{}, {}]
    total_test = [{}, {}]

    for classe in range(2):
        print(f"Counting occurrences of word combinations for class {classe}...")
        for s, semelhantes_list in tqdm(semelhantes[classe].items()):
            # Extract sentences from similar lists
            lista_sentencas = [sublista[0] for sublista in semelhantes_list]

            # Find word combinations and generate pairs and triplets
            combinacoes_encontradas = encontrar_combinacoes(s, fi[classe])
            lista_combinacoes = combinacoes_encontradas.copy()
            lista_combinacoes.extend([' '.join(comb) for comb in combinations(combinacoes_encontradas, 2)])
            lista_combinacoes.extend([' '.join(comb) for comb in combinations(combinacoes_encontradas, 3)])

            # Count occurrences in similar sentences and the test sentence
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

        # Filter and sort training data combinations
        total_treino[classe] = {chave: valor for chave, valor in total_treino[classe].items() if valor >= 2}
        total_treino[classe] = dict(sorted(total_treino[classe].items(), key=lambda item: item[1], reverse=True))

        # Sort test data combinations
        total_test[classe] = dict(sorted(total_test[classe].items(), key=lambda item: item[1], reverse=True))

        # Save results to JSON files
        io.gravarJson(os.path.join(diretorio, fold, p.p_inf, f'{fold}{md_name.upper()}ErrorTreino{classe}{sent.capitalize()}.json'), dict(total_treino[classe]))
        io.gravarJson(os.path.join(diretorio, fold, p.p_inf, f'{fold}{md_name.upper()}ErrorTeste{classe}{sent.capitalize()}.json'), dict(total_test[classe]))

    return [total_treino, total_test]


def recuperarSentencasSemelhantesTreinoPorClasse(diretorio, fold, md_name, outro=True):
    """
    Retrieve the most similar training sentences for error samples by class using BERT embeddings.

    Parameters:
    - diretorio: Directory containing the data.
    - fold: Fold identifier for cross-validation.
    - md_name: Model name (e.g., BERT).
    - outro: Boolean indicating whether to retrieve sentences from the opposite class (True) or the same class (False).

    Returns:
    - Saves JSON files containing the most similar sentences for each class and sample.
    """
    # Load training and error data
    print("Loading data...")
    df_treino = pd.read_csv(os.path.join(diretorio, fold, f'{fold}Treino.csv'))
    df_erros = pd.read_csv(os.path.join(diretorio, fold, 'misclassified_train.csv'))
    lfe = [(df_erros[df_erros['classe_real'] == 0])['texto'].to_list(),
           (df_erros[df_erros['classe_real'] == 1])['texto'].to_list()]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load BERT model and tokenizer
    print("Loading BERT model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    model.eval()

    # Determine training sentences for each class based on the "outro" parameter
    if outro:
        lft = [(df_treino[df_treino['class_label'] == 1])['text_pp'].to_list(),
               (df_treino[df_treino['class_label'] == 0])['text_pp'].to_list()]
    else:
        lft = [(df_treino[df_treino['class_label'] == 0])['text_pp'].to_list(),
               (df_treino[df_treino['class_label'] == 1])['text_pp'].to_list()]

    semelhantes = [dict(), dict()]

    for classe in [0, 1]:
        sentencesErro = lfe[classe]
        sentencesTreino = lft[classe]

        # Generate embeddings for training sentences
        embeddingsTreino = []
        for sentence in tqdm(sentencesTreino, desc=f"Generating embeddings for training sentences of class {classe}", colour='yellow'):
            inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs.to(device))
            embeddingsTreino.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())

        embeddingsTreino_np = np.array(embeddingsTreino)

        # Create FAISS index for similarity search
        index = faiss.IndexFlatIP(embeddingsTreino_np.shape[1])
        faiss.normalize_L2(embeddingsTreino_np)
        index.add(embeddingsTreino_np)

        # Generate embeddings for error sentences
        embeddingsErro = []
        for sentence in tqdm(sentencesErro, desc=f"Generating embeddings for error sentences of class {classe}", colour='yellow'):
            inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs.to(device))
            embeddingsErro.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())

        embeddingsErro_np = np.array(embeddingsErro)
        faiss.normalize_L2(embeddingsErro_np)

        # Find most similar training sentences for each error sentence
        for i, sentence in enumerate(tqdm(sentencesErro, desc=f"Searching for similar sentences for class {classe}", colour='yellow')):
            query_vector = embeddingsErro_np[i].reshape(1, -1)

            distances, indices = index.search(query_vector, len(sentencesTreino))

            # Retrieve and sort sentences based on similarity
            most_similar_sentences = [(sentencesTreino[idx], float(dist)) for idx, dist in zip(indices[0], distances[0]) if dist >= 0.7]
            most_similar_sentences.sort(key=lambda x: x[1], reverse=True)

            semelhantes[classe][sentence] = most_similar_sentences[:100]

        print("Saving results...")
        if outro:
            io.gravarJson(os.path.join(diretorio, fold, p.p_inf, f'{fold}{md_name}Semelhantes{classe}Outro.json'), semelhantes[classe])
        else:
            io.gravarJson(os.path.join(diretorio, fold, p.p_inf, f'{fold}{md_name}Semelhantes{classe}Mesmo.json'), semelhantes[classe])


def calcularConf(diretorio: str, fold: int):
    """
    Compute the frequencies of error patterns and generate CSV files for analysis.

    Parameters:
    - diretorio: Directory containing the data.
    - fold: Fold identifier for cross-validation.

    This function performs the following steps:
    1. Retrieve the most similar training sentences for error samples for both opposite and same classes.
    2. Count occurrences of word combinations in similar sentences for both opposite and same classes.
    3. Generate and save CSV files containing frequencies of word combinations for analysis.
    """
    print('Calculating error pattern frequencies...')

    # Step 1: Retrieve similar training sentences for opposite and same classes
    recuperarSentencasSemelhantesTreinoPorClasse(diretorio, fold, 'bert', outro=True)
    recuperarSentencasSemelhantesTreinoPorClasse(diretorio, fold, 'bert', outro=False)

    print("Counting similar error occurrences...")

    # Step 2: Count occurrences of word combinations
    outro = contar_ocorrencias_semelantesErro(diretorio, fold, 'bert', 'outro')
    mesmo = contar_ocorrencias_semelantesErro(diretorio, fold, 'bert', 'mesmo')

    print("Creating CSV files...")

    # Step 3: Generate and save CSV files for opposite and same classes
    csv_path = os.path.join(diretorio, fold, p.p_inf, f'{fold}bertErrorFeatures1.csv')
    create_frequency_csv(outro[1][0], mesmo[0][0], outro[0][0], csv_path)

    csv_path = os.path.join(diretorio, fold, p.p_inf, f'{fold}bertErrorFeatures0.csv')
    create_frequency_csv(outro[1][1], mesmo[0][1], outro[0][1], csv_path)

    # Update occurrences of test errors for same class
    contar_ocorrencias_semelantes_teste_mesmo(diretorio, fold, 'bert')
