import sys
import nltk

from nltk.probability import FreqDist
# Adding Folder_2/subfolder to the system path
sys.path.insert(1, '../')

from utils import LimeUtils as u
from utils import io as io

import pandas

import pandas as pd
import statistics
from operator import itemgetter
from pathlib import Path

from os.path import exists

import numpy as np
import matplotlib.pyplot as plt
import json

from collections import Counter
# dictionary = Counter(filtered_text)
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import glob, os

from joblib import dump, load
from nltk.probability import FreqDist, ConditionalFreqDist

import seaborn as sns

from utils import models as md
from utils import path as p
from utils import importantFeatures as fi
from utils import io as io


# Constants definition
data_path = './datasets/if/'
class_names = ['Other acquisitions', 'Specific acquisitions']
num_features = 500


def createWords(features):
    """
    Create a sorted list of unique words from the features.

    Args:
        features (list): List of tuples containing words and their values.

    Returns:
        list: Sorted list of unique words.
    """
    words = []
    for tupla in features:
        word = tupla[0]
        if word not in words:
            words.append(word)
    return sorted(words)  


def createEmptyVocab(features):
    """
    Create an empty vocabulary dictionary with words as keys.

    Args:
        features (list): List of tuples containing words and their values.

    Returns:
        dict: Dictionary with words as keys and empty lists as values.
    """
    vocab = dict()
    for tupla in features:
        word = tupla[0]
        vocab[word] = []
               
    return vocab


def createValues(features):
    """
    Create a vocabulary dictionary with words as keys and their values as lists.

    Args:
        features (list): List of tuples containing words and their values.

    Returns:
        dict: Dictionary with words as keys and lists of values.
    """
    vocab = createEmptyVocab(features)
    for tupla in features:
        word  = tupla[0]
        value = tupla[1]
        vocab[word].append(value)
    return vocab


def calcListMeans(features):
    """
    Calculate the mean of values for each word.

    Args:
        features (list): List of tuples containing words and their values.

    Returns:
        list: List of mean values for each word.
    """
    words = createWords(features)
    vocab = createValues(features)
    listMeans = [0] * len(words)
    for key in vocab:
        mean = statistics.mean(vocab[key])
        i = words.index(key)
        listMeans[i] = mean

    return listMeans


def createDataFrame(features):
    """
    Create a DataFrame with words as columns and their mean values as rows.

    Args:
        features (list): List of tuples containing words and their values.

    Returns:
        DataFrame: DataFrame with words as columns and mean values as rows.
    """
    words = createWords(features)
    vocab = createValues(features)
    record = [0] * len(words)
    values = dict()
    for key in vocab:
        mean = statistics.mean(vocab[key])
        i = words.index(key)
        record[i] = mean

    df = pd.DataFrame([record], columns=words)    
    return df


def intersec(a1, b1):
    """
    Calculate the intersection of two lists.

    Args:
        a1 (list): First list.
        b1 (list): Second list.

    Returns:
        set: Intersection of the two lists.
    """
    a = set(a1)
    b = set(b1)
    return a.intersection(b)


def calcDictSqrt(features):
    """
    Calculate the square root of the sum of values for each word.

    Args:
        features (list): List of tuples containing words and their values.

    Returns:
        dict: Dictionary with words as keys and square root of sum of values as values.
    """
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


def calcDictMeans(features):
    """
    Calculate the mean of values for each word.

    Args:
        features (list): List of tuples containing words and their values.

    Returns:
        dict: Dictionary with words as keys and mean of values as values.
    """
    words = createWords(features)
    vocab = createValues(features)
    dicMeans = dict()
    for key in vocab:
        mean = statistics.mean(vocab[key])
        dicMeans[key] = mean

    return dicMeans


def calcDictSum(features):
    """
    Calculate the sum of values for each word.

    Args:
        features (list): List of tuples containing words and their values.

    Returns:
        dict: Dictionary with words as keys and sum of values as values.
    """
    words = createWords(features)
    vocab = createValues(features)
    dicMeans = dict()
    for key in vocab:
        mean = np.sum(vocab[key])
        dicMeans[key] = mean

    return dicMeans


def normalize_dictionary_values(dictionary):
    """
    Normalize the values of a dictionary.

    Args:
        dictionary (dict): Dictionary with values to be normalized.

    Returns:
        dict: Dictionary with normalized values.
    """
    # Get the maximum and minimum values from the dictionary
    values = list(dictionary.values())
    max_value = max(values)
    min_value = min(values)

    # Normalize the values
    normalized_dict = {}
    for key, value in dictionary.items():
        max_value_min_value = 0.0000000001 if (max_value - min_value) == 0 else max_value - min_value
        normalized_value = (value - min_value) / (max_value_min_value)
        normalized_dict[key] = normalized_value

    return normalized_dict

# Load Lime files
def loadLime(path, prefix, model_name):
    """
    Load Lime feature importance files and normalize their values.

    Args:
        path (str): Base path to the files.
        prefix (str): Prefix for the file names.
        model_name (str): Name of the model.

    Returns:
        dict: Dictionary with model names as keys and normalized feature importance values.
    """
    mapFeaturesModel = dict()
    path_if = f'{path}{p.path_if}'

    mapMeans = dict()
    model_name
    file_name0 = f'{path_if}{prefix}{model_name}{io.arquivo_if}0.json'
    if exists(file_name0):
        features = io.lerJson(file_name0)
        mapMeans[0] = calcDictSqrt(features)
        mapMeans[0] = normalize_dictionary_values(mapMeans[0])
    else:
        print(f'{file_name0} does not exist')

    file_name1 = f'{path_if}{prefix}{model_name}{io.arquivo_if}1.json'
    if exists(file_name1):
        features = io.lerJson(file_name1)
        mapMeans[1] = calcDictSqrt(features)
        mapMeans[1] = normalize_dictionary_values(mapMeans[1])

    mapFeaturesModel[model_name] = mapMeans

    return mapFeaturesModel

# Load Shape files
def loadShape(path, prefix):
    """
    Load Shape feature importance files and normalize their values.

    Args:
        path (str): Base path to the files.
        prefix (str): Prefix for the file names.

    Returns:
        dict: Dictionary with model names as keys and normalized feature importance values.
    """
    map_features = dict()
    path_if = f'{path}if/'
    for model_name in md.bestClassifiers:
        model_name_ef = md.models_names_ef[model_name]
        file_name = f'{path_if}{model_name_ef}{prefix}_Vals.shape'
        if exists(file_name):
            classifier_name = file_name[0:file_name.index('_')]
            if_values = load(file_name)

            map_class = dict()
            map_if0 = dict()
            map_if1 = dict()

            map_if0 = {key: values[0] for key, values in if_values.items() if values[0] != 0 and values[1] != 0 and values[0] > values[1]}
            map_if0 = normalize_dictionary_values(map_if0)
            map_if1 = {key: values[1] for key, values in if_values.items() if values[0] != 0 and values[1] != 0 and values[0] < values[1]}
            map_if1 = normalize_dictionary_values(map_if1)

            map_if0 = dict(sorted(map_if0.items(), key=itemgetter(1), reverse=True))
            map_if1 = dict(sorted(map_if1.items(), key=itemgetter(1), reverse=True))

            map_class[0] = map_if0
            map_class[1] = map_if1

            map_features[model_name_ef] = map_class

    return map_features

# Load all files with important features
def loadImportantFeatures(path, prefix, model_name):
    """
    Load important features from both Lime and Shape files.

    Args:
        path (str): Base path to the files.
        prefix (str): Prefix for the file names.
        model_name (str): Name of the model.

    Returns:
        tuple: Two dictionaries containing important features from Lime and Shape.
    """
    mapFeaturesLime = loadLime(path, prefix, model_name)
    mapFeaturesShape = loadShape(path, prefix)
    return mapFeaturesLime, mapFeaturesShape

def features_cl(mapFeaturesLime, mapFeaturesShape, cl, num_features):
    """
    Generate a string with the top features from Lime and Shape for a given class.

    Args:
        mapFeaturesLime (dict): Dictionary with Lime features.
        mapFeaturesShape (dict): Dictionary with Shape features.
        cl (int): Class index.
        num_features (int): Number of top features to include.

    Returns:
        str: String with the top features and their values.
    """
    result = ''
    sorted_map_shap = sorted(mapFeaturesShape['SgdTfIdf'][cl].items(), key=lambda x: x[1], reverse=True)
    sorted_map_lime = sorted(mapFeaturesLime['SgdTfIdf'][cl].items(), key=lambda x: x[1], reverse=True)
    for i in range(0, num_features):
        tuple_lime = sorted_map_lime[i]
        tuple_shape = sorted_map_shap[i]
        result = result + f'{cl},{tuple_lime[0]},{np.round(tuple_lime[1], 3)},{tuple_shape[0]},{np.round(tuple_shape[1], 3)}\n'
    return result

def features(dts, ite, clf, num_features):
    """
    Generate a CSV file with the top features from Lime and Shape for all classes.

    Args:
        dts (str): Dataset name.
        ite (str): Iteration name.
        clf (str): Classifier name.
        num_features (int): Number of top features to include.

    Returns:
        None
    """
    prefix = f'{dts}{ite}'
    model_name = md.models[clf]

    path = f'{p.path_dts}{model_name}/{dts}/'
    mapFeaturesLime, mapFeaturesShape = loadImportantFeatures(path, prefix, model_name)
    result = 'cl,lime,val_lime,shape,val_shape\n'

    result = result + features_cl(mapFeaturesLime, mapFeaturesShape, 0, num_features)
    result = result + features_cl(mapFeaturesLime, mapFeaturesShape, 1, num_features)
    io.saveFile(path, f'\out\{prefix}fi.csv', result)
    

    
def lerIf(path, prefix, model_name, explainer):
    """
    Load important features and save them in a summarized format.

    Args:
        path (str): Base path to the files.
        prefix (str): Prefix for the file names.
        model_name (str): Name of the model.
        explainer (str): Type of explainer ('Lime' or 'Shape').

    Returns:
        list: List containing dictionaries of important features for each class.
    """
    # Load important features
    mapFeaturesLime = dict()
    mapFeaturesShape = dict()
    mapFeaturesLime, mapFeaturesShape = fi.loadImportanteFeatures(path, prefix, model_name)
    clf = f'{model_name}'

    mapFeatures = mapFeaturesLime
    if explainer == 'Shape':
        mapFeatures = mapFeaturesShape

    # Load features for the specific classifier
    f0 = mapFeatures[clf][0]
    f0 = dict(sorted(f0.items(), key=itemgetter(1), reverse=False))
    f0 = {key: value for key, value in f0.items()}

    f1 = mapFeatures[clf][1]
    f1 = dict(sorted(f1.items(), key=itemgetter(1), reverse=False))
    f1 = {key: value for key, value in f1.items()}

    f_global = [f0, f1]

    f0o = dict(sorted(f0.items(), key=itemgetter(1), reverse=True))
    f0o = {key: value for key, value in f0o.items()}

    f1o = dict(sorted(f1.items(), key=itemgetter(1), reverse=True))
    f1o = {key: value for key, value in f1o.items()}

    io.gravarJson(f'{path}{p.path_if}{prefix}{model_name}{io.arquivo_if_resumido}0.json', f0o)
    io.gravarJson(f'{path}{p.path_if}{prefix}{model_name}{io.arquivo_if_resumido}1.json', f1o)

    return f_global
