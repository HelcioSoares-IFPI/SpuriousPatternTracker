"""
This script computes global explanations using LIME for a classification model trained on text data.
Key functionalities include:
- Predicting model outputs in batches to optimize GPU memory usage.
- Generating global explanations for training data using LIME.
- Saving and loading explanation results to/from JSON files for efficient computation.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from lime.lime_text import LimeTextExplainer
from torch.nn.functional import softmax
import pandas as pd
import gc
from tqdm.auto import tqdm
import os
import numpy as np
from utils.utils import save_dict_to_json, load_dict_from_json
from utils.path import verificarCaminho


def model_predictor(texts: list, batch_size=4):
    """
    Predicts model outputs in batches to optimize GPU memory usage.

    Args:
        texts (list of str): List of texts for prediction.
        batch_size (int): Batch size for processing.

    Returns:
        numpy.array: Predicted probabilities for all texts.
    """
    all_probabilities = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            inputs = inputs.to(device)
            logits = model(**inputs).logits
            probabilities = softmax(logits, dim=1).cpu().numpy()
            all_probabilities.append(probabilities)
        torch.cuda.empty_cache()
    return np.concatenate(all_probabilities, axis=0)


def compute_global_explanation(ultimo: int, train_texts: list, fold: int, model_predictor: list, diretorio: str, num_samples=500, num_explanations=None):
    """
    Computes global explanations using LIME based on training sentences.

    Args:
        ultimo (int): Index to resume computation from.
        train_texts (list): List of training sentences.
        fold (int): Fold number being processed.
        model_predictor (list): Model prediction function.
        diretorio (str): Directory to save explanation results.
        num_samples (int): Number of samples to generate for LIME explanation.
        num_explanations (int, optional): Number of explanations to compute. Defaults to the length of `train_texts`.

    Returns:
        None
    """
    explainer = LimeTextExplainer(class_names=['General acquisitions', 'Specific acquisitions'])
    diretorioExp = os.path.join(diretorio, 'exp')

    if os.path.exists(os.path.join(diretorioExp, 'allLocalExp.json')):
        global_weights = load_dict_from_json(os.path.join(diretorioExp, 'allLocalExp.json'))
    else:
        global_weights = {}

    if num_explanations is None:
        num_explanations = len(train_texts)

    for i, text in tqdm(enumerate(train_texts), total=num_explanations, desc="Computing global explanation for fold {}".format(fold), colour="green"):
        if (i + 1) % 100 == 0:  # Save every 100 iterations
            save_dict_to_json([ultimo + i], os.path.join(diretorioExp, 'ultimo.json'))
        exp = explainer.explain_instance(text, model_predictor, labels=[0, 1], num_features=60, num_samples=num_samples)
        global_weights[text] = dict(exp.as_list())

        del exp
        gc.collect()
        if (i + 1) % 100 == 0:  # Save every 100 iterations
            save_dict_to_json(global_weights, os.path.join(diretorioExp, 'allLocalExp.json'))
    del explainer
    gc.collect()
    return


def explicarLimeGlobal(diretorio: str, fold: int) -> None:
    """
    Generates global LIME explanations for the specified fold.

    Args:
        diretorio (str): Directory containing the fold data.
        fold (int): Fold number being processed.

    Returns:
        None
    """
    print('\nGenerating Global LIME Explanations for fold: {}\n'.format(fold))
    global device
    device = torch.device("cuda")

    diretorioFold = os.path.join(diretorio, str(fold))
    valid_text = pd.read_csv(os.path.join(diretorioFold, '{}Treino_.csv'.format(fold)))['texto'].to_list()

    diretorioModel = os.path.join(diretorioFold, 'model')
    global model
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(diretorioModel)).to(device)
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(diretorioModel)

    diretorioExp = os.path.join(diretorioFold, 'exp')
    verificarCaminho(diretorioExp)
    
    if os.path.exists(os.path.join(diretorioExp, 'ultimo.json')):
        print('Loading the last saved index.')
        ultimo = load_dict_from_json(os.path.join(diretorioExp, 'ultimo.json'))[0]
    else:
        print('No saved index found. Starting from zero.')
        ultimo = 0

    train_texts_sample = valid_text[ultimo:]
    compute_global_explanation(ultimo, train_texts_sample, fold, model_predictor, diretorioFold)
