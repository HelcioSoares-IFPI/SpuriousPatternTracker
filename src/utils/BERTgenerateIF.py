"""
This script processes feature importance (FI) from textual data and generates combinations of important features.
Key functionalities include:
- Normalizing feature importance values.
- Filtering and processing features into interpretable formats.
- Generating combinations of n-grams based on important features.
- Verifying feature presence in test sentences.
- Saving results to JSON files for future analysis.
"""

import numpy as np
import os
from typing import *
import pandas as pd
from itertools import combinations
from tqdm.auto import tqdm
from utils.preProcessamento import remover_stopwords, pre_processa_palavra, preprocessar_lista_frases
from utils.path import verificarCaminho
from utils.utils import save_dict_to_json, load_dict_from_json


def normalize_dictionary_values(dictionary: dict) -> dict:
    """
    Normalizes the values of a dictionary to a [0, 1] range.

    Parameters:
    - dictionary (dict): Input dictionary with numerical values.

    Returns:
    - dict: Dictionary with normalized values.
    """
    values = list(dictionary.values())
    max_value = max(values)
    min_value = min(values)
    range_value = max_value - min_value or 0.1  # Avoid division by zero
    return {key: (value - min_value) / range_value for key, value in dictionary.items()}


def process_features(features: list) -> dict:
    """
    Processes features to calculate their aggregate importance using L2 norm.

    Parameters:
    - features (list): List of tuples (word, importance value).

    Returns:
    - dict: Dictionary with words as keys and their aggregated importance values.
    """
    vocab = {}
    for word, value in features:
        vocab.setdefault(word, []).append(value)
    return {word: np.sqrt(np.sum(np.array(values)**2)) for word, values in vocab.items()}


def load_and_normalize_features(features: list) -> dict:
    """
    Loads and normalizes feature importance values.

    Parameters:
    - features (list): List of tuples (word, importance value).

    Returns:
    - dict: Dictionary with normalized feature importance values.
    """
    processed_features = process_features(features)
    normalized_features = normalize_dictionary_values(processed_features)
    return normalized_features


def processIF(words: list) -> dict:
    """
    Processes word features by removing stopwords and sorting them by importance.

    Parameters:
    - words (list): List of tuples (word, importance value).

    Returns:
    - dict: Dictionary of processed features sorted by importance.
    """
    fi = remover_stopwords(load_and_normalize_features(words))
    fi = {key: value for key, value in fi.items() if key not in ['', ' ']}
    fi = dict(sorted(fi.items(), key=lambda item: item[1], reverse=True))
    return fi


def criarIF(fold: int, diretorioExp: str, diretorioIF: str) -> list:
    """
    Creates feature importance (IF) files from the allLocalExp.json file.

    Parameters:
    - fold (int): Current fold number.
    - diretorioExp (str): Directory containing explanation results.
    - diretorioIF (str): Directory to save the IF files.

    Returns:
    - list: Two dictionaries containing positive and negative feature importances.
    """
    data = load_dict_from_json(os.path.join(diretorioExp, 'allLocalExp.json'))
    negative_words, positive_words = [], []

    for sentence, word_values in data.items():
        for word, value in word_values.items():
            word = pre_processa_palavra(word)
            if value < 0:
                negative_words.append((word, abs(value)))
            else:
                positive_words.append((word, value))

    fi0 = processIF(negative_words)
    fi1 = processIF(positive_words)

    # Save the lists as JSON files
    save_dict_to_json(fi0, os.path.join(diretorioIF, '{}0bert0.json'.format(fold)))
    save_dict_to_json(fi1, os.path.join(diretorioIF, '{}0bert1.json'.format(fold)))

    return [fi0, fi1]


def palavra_existe_nas_sentencas(sentenca: str, lista_sentencas: list) -> bool:
    """
    Checks if all words in a sentence exist in any sentence from a list, regardless of order.

    Parameters:
    - sentenca (str): The sentence to check.
    - lista_sentencas (list): List of sentences to verify against.

    Returns:
    - bool: True if all words exist in any sentence from the list, False otherwise.
    """
    words_in_sentence = set(sentenca.split())
    for sentence in lista_sentencas:
        words_in_sentence_list = set(sentence.split())
        if words_in_sentence.issubset(words_in_sentence_list):
            return True
    return False


def gerar_combinacoes(palavras: list, n: int, sentencas_teste: list) -> list:
    """
    Generates n-gram combinations of words and filters them by presence in test sentences.

    Parameters:
    - palavras (list): List of words for combination.
    - n (int): Maximum number of words in a combination.
    - sentencas_teste (list): List of test sentences for filtering.

    Returns:
    - list: List of valid combinations.
    """
    n_gram_combinations = []
    for i in tqdm(range(2, n + 1), desc='Generating combinations', colour='yellow'):
        combinations_list = [' '.join(comb) for comb in combinations(palavras, i)]
        n_gram_combinations.extend(combinations_list)
    filtered_combinations = [comb for comb in tqdm(n_gram_combinations, desc='Filtering combinations', colour='blue') if palavra_existe_nas_sentencas(comb, sentencas_teste)]
    all_combinations = []
    all_combinations.extend(palavras)
    all_combinations.extend(filtered_combinations)
    return all_combinations


def gerarIF(diretorio: str, fold: int) -> None:
    """
    Generates feature importance (IF) combinations for a specific fold.

    Parameters:
    - diretorio (str): Directory containing the data for the fold.
    - fold (int): Current fold number.

    Returns:
    - None
    """
    n = 3  # Maximum n-gram size for combinations

    print('\nGenerating IF for fold {}...\n'.format(fold))
    diretorioFold = os.path.join(diretorio, str(fold))
    verificarCaminho(diretorioFold)
    diretorioExp = os.path.join(diretorioFold, 'exp')
    verificarCaminho(diretorioExp)
    diretorioIf = os.path.join(diretorioFold, 'if')
    verificarCaminho(diretorioIf)

    test_sentences = pd.read_csv(os.path.join(diretorioFold, '{}Teste_.csv'.format(fold)))['texto'].to_list()
    test_sentences = preprocessar_lista_frases(test_sentences)
    features = criarIF(fold, diretorioExp, diretorioIf)

    # Generate combinations for negative features
    fis0 = list(features[0].keys())[0:60]
    res0 = gerar_combinacoes(fis0, n, test_sentences)

    # Generate combinations for positive features
    fis1 = list(features[1].keys())[0:60]
    res1 = gerar_combinacoes(fis1, n, test_sentences)

    # Save the results as JSON files
    save_dict_to_json([res0, res1], os.path.join(diretorioIf, '{}bertCombs.json'.format(fold)))
