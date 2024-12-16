"""
Fold Generator Script for SpuriousPatternTracker

This script is part of the SpuriousPatternTracker pipeline. It preprocesses datasets, creates error tracking files, 
and generates k-fold splits for training and testing machine learning models. The folds are used in the subsequent 
steps of the method.

Dependencies:
- pandas
- utils (preProcessamento, io, path, models)
"""

import sys
# Add utility folder to the system path
sys.path.insert(1, '../')
from utils import preProcessamento as pp
from utils import io
from utils import path as p
from utils import models as md
import pandas as pd


def createDatasetPadrao(output_path, class_label):
    """
    Preprocess the input dataset and create a standardized CSV file.

    Args:
        output_path (str): The directory where the output file will be saved.
        class_label (int): The class label for filtering (0 or 1).
    """
    # Load the source data (the file should contain 'text' and 'class' columns)
    df = pd.read_csv(f'arquivo de origem')  # Replace with the actual source file path
    textos = df['text'].tolist()  # Raw text
    class_labels = df['class'].tolist()  # Class labels (0 or 1)
    text_pp = pp.preProcessaTexto03(textos.copy())  # Preprocess the text
    tipo = [0] * len(text_pp)  # Placeholder column
    ite = [0] * len(text_pp)  # Placeholder column

    # Create the output DataFrame
    columns = ['text', 'text_pp', 'class_label']
    tuples = list(zip(textos, text_pp, class_labels, tipo, ite))
    df_out = pd.DataFrame(tuples, columns=columns)

    # Save the processed data
    output_file = f'{output_path}classe{class_label}.csv'
    df_out.to_csv(output_file, encoding='utf-8', sep=',', index=False)


def create_error_tracking_file(model_name):
    """
    Create an empty CSV file to track errors for a specific model.

    Args:
        model_name (str): The name of the model.
    """
    df = pd.DataFrame(columns=['text_pp', 'class_label', 'text'])
    file_name = f'{model_name}Erros.csv'
    df.to_csv(file_name, index=False)


def create_folds(model, k):
    """
    Generate k-fold splits for a specific model.

    Args:
        model (str): The model name.
        k (int): Number of folds.
    """
    # Load the class-specific datasets
    positive_data = pd.read_csv(f'{p.path_dts}classe1.csv')
    negative_data = pd.read_csv(f'{p.path_dts}classe0.csv')

    # Determine fold sizes
    len_fold_pos = len(positive_data) // k
    len_fold_neg = len(negative_data) // k

    # Generate folds
    for fold in range(k):
        path, prefix, model_name = io.createCaminho(model, fold + 1)

        # Define fold ranges
        start_pos, end_pos = fold * len_fold_pos, (fold + 1) * len_fold_pos
        start_neg, end_neg = fold * len_fold_neg, (fold + 1) * len_fold_neg

        # Create test and train splits
        test_pos = positive_data.iloc[start_pos:end_pos]
        test_neg = negative_data.iloc[start_neg:end_neg]
        train_pos = pd.concat([positive_data.iloc[:start_pos], positive_data.iloc[end_pos:]])
        train_neg = pd.concat([negative_data.iloc[:start_neg], negative_data.iloc[end_neg:]])

        train_data = pd.concat([train_pos, train_neg]).reset_index(drop=True)
        test_data = pd.concat([test_pos, test_neg]).reset_index(drop=True)

        # Save the train and test datasets
        io.createPasta(path)
        train_data.to_csv(f'{path}{prefix}Treino.csv', index=False)
        test_data.to_csv(f'{path}{prefix}Teste.csv', index=False)


# Main program
if __name__ == "__main__":
    # Step 1: Create standardized datasets for each class
    createDatasetPadrao(p.path_dts, 0)
    createDatasetPadrao(p.path_dts, 1)

    # Step 2: Create error tracking files for all models
    for model in md.models.keys():
        model_name = md.models[model]
        create_error_tracking_file(model_name)

    # Step 3: Generate k-fold splits for each model
    k_folds = 5  # Number of folds
    for model in md.models.keys():
        for k in range(1, k_folds + 1):
            create_folds(model, k)
