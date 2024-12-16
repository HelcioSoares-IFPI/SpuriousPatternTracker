import json
import pandas as pd
from utils import path as p
from utils import preProcessamento as pp
from utils import models as md
import os
import fnmatch

delimiter = ','
ponto_virgula = ';'

# File names
arquivo_if = 'ListClass'
arquivo_er = 'ErrorFeatures'
arquivo_ce = 'CEnn'

arquivo_ce_resumido = 'CE'
arquivo_if_resumido = 'IF'
arquivo_er_resumido = 'EF'

arquivo_ex = 'allLocalExp_tr_ts'

def gravarJson(nomeArquivo, obj):
    """
    Save an object as a JSON file.

    Args:
        nomeArquivo (str): File name.
        obj (object): Object to save.

    Returns:
        None
    """
    with open(nomeArquivo, 'w', encoding='UTF-8') as f:
        str = json.dumps(obj)
        f.write(str)
    
def lerJson(nomeArquivo):
    """
    Load an object from a JSON file.

    Args:
        nomeArquivo (str): File name.

    Returns:
        object: Loaded object.
    """
    with open(nomeArquivo, encoding='UTF-8') as f:
        s = f.read()
        data = json.loads(s)        
    return data

def saveFile(path, fileName, lista):
    """
    Save a list to a file.

    Args:
        path (str): File path.
        fileName (str): File name.
        lista (list): List to save.

    Returns:
        None
    """
    with open(path + fileName, 'w', encoding='UTF-8') as f:
        f.write(lista)

def printImputsLen(df, X, y, class_names, type_):
    """
    Print the length of inputs.

    Args:
        df (DataFrame): DataFrame containing the data.
        X (list): List of input texts.
        y (list): List of labels.
        class_names (list): List of class names.
        type_ (str): Type of data (e.g., 'Train', 'Test').

    Returns:
        None
    """
    print(type_ + ':', len(X))
    #print('Class names:', class_names)
    
def createInputs(file_path, file_name, sep, type_):
    """
    Create inputs for the classifier.

    Args:
        file_path (str): File path.
        file_name (str): File name.
        sep (str): Separator used in the file.
        type_ (str): Type of data (e.g., 'Train', 'Test').

    Returns:
        tuple: Tuple containing texts, preprocessed texts, labels, class names, types, and iterations.
    """
    df = pd.read_csv(file_path + file_name, sep=sep)
    X_ = df["text_pp"].tolist()
    y_ = df["class_label"].tolist()
    t_ = df["text"].tolist()
    class_names = ['Não aquisição', 'Aquisição']
    tipo_ = df["tipo"].tolist()
    ite = df["ite"].tolist()

    #printImputsLen(df, X_, y_, class_names, type_)
    X_ = pp.preProcessaTexto03(X_)
    return t_, X_, y_, class_names, tipo_, ite

def createInputs_ite(file_path, file_name, sep, type_):
    """
    Create inputs for the classifier with iterations.

    Args:
        file_path (str): File path.
        file_name (str): File name.
        sep (str): Separator used in the file.
        type_ (str): Type of data (e.g., 'Train', 'Test').

    Returns:
        tuple: Tuple containing texts, preprocessed texts, labels, class names, types, and iterations.
    """
    df = pd.read_csv(file_path + file_name, sep=sep)
    X_ = df["text_pp"].tolist()
    y_ = df["class_label"].tolist()
    t_ = df["text"].tolist()
    class_names = ['Não aquisição', 'Aquisição']
    tipo_ = df["tipo"].tolist()
    ite = df["ite"].tolist()
    #printImputsLen(df, X_, y_, class_names, type_)
    #X_ = pp.preProcessaTexto01(X_)
    return t_, X_, y_, class_names, tipo_, ite
    
def loadDataset(caminho, pref, tipo):
    """
    Load a dataset.

    Args:
        caminho (str): Base path to the files.
        pref (str): Prefix for the file names.
        tipo (str): Type of data (e.g., 'Train', 'Test').

    Returns:
        tuple: Tuple containing texts, preprocessed texts, labels, class names, types, and iterations.
    """
    #print(f'{caminho}{tipo}.csv')
    df = pd.read_csv(f'{caminho}{pref}{tipo}.csv', sep=delimiter)
    X_ = df["text_pp"].tolist()
    y_ = df["class_label"].tolist()
    t_ = df["text"].tolist()
    class_names = ['Não aquisição', 'Aquisição']
    tipo_ = df["tipo"].tolist()
    ite = df["ite"].tolist()
  
    X_ = pp.preProcessaTexto01(X_)
    return t_, X_, y_, class_names, tipo_, ite
    
def createqDataset(lista_de_tuplas, labels, path_datasets, datasetName):
    """
    Create a dataset from a list of tuples.

    Args:
        lista_de_tuplas (list): List of tuples containing the data.
        labels (list): List of column labels.
        path_datasets (str): Path to save the dataset.
        datasetName (str): Name of the dataset file.

    Returns:
        None
    """
    df_out = pd.DataFrame(lista_de_tuplas, columns=labels)    
    df_out.to_csv(path_datasets + datasetName, encoding='utf-8', sep=delimiter, index=False)
    

def loadDatasets(pre):
    """
    Load training and test datasets.

    Args:
        pre (str): Prefix for the file names.

    Returns:
        tuple: Tuple containing training and test data.
    """
    X_train, y_train, class_names = createInputs(p.path_dts, f'{pre}Treino.csv', delimiter, 'Treino')
    X_test, y_test, class_names = createInputs(p.path_dts, f'{pre}Teste.csv', delimiter, 'Teste')
    return X_train, y_train, X_test, y_test, class_names


def loadText(file_path, file_name, sep):
    """
    Load text data from a file.

    Args:
        file_path (str): File path.
        file_name (str): File name.
        sep (str): Separator used in the file.

    Returns:
        DataFrame: DataFrame containing the text data.
    """
    df = pd.read_csv(file_path + file_name, sep=sep)
    return df 
    
def loadDatasets_ite(pre, caminho):
    """
    Load datasets with iterations.

    Args:
        pre (str): Prefix for the file names.
        caminho (str): Base path to the files.

    Returns:
        tuple: Tuple containing training and test data with iterations.
    """
    # Create inputs for the classifier
    X_text_tr, X_train, y_train, class_names, tipo_train, ite_train = createInputs_ite(caminho, f'{pre}Treino.csv', delimiter, 'Treino')
    X_text_ts, X_test, y_test, class_names, tipo_test, ite_test = createInputs_ite(caminho, f'{pre}Teste.csv', delimiter, 'Teste')
    return X_text_tr, X_train, y_train, tipo_train, ite_train, X_text_ts, X_test, y_test, class_names, tipo_test, ite_test
    
def loadDatasets_ite01(pre, caminho):
    """
    Load datasets for iteration 01.

    Args:
        pre (str): Prefix for the file names.
        caminho (str): Base path to the files.

    Returns:
        tuple: Tuple containing training and test data.
    """
    #print(ite, caminho)
    # Create inputs for the classifier
    X_text_tr, X_train, y_train, class_names, tipo_train, ite_train = createInputs_ite(caminho, f'{pre}Treino.csv', delimiter, 'Treino')
    X_text_ts, X_test, y_test, class_names, tipo_test, ite_test = createInputs_ite(caminho, f'{pre}Teste.csv', delimiter, 'Teste')
    return X_train, y_train, X_test, y_test    

def loadDatasets_ite02(pre, caminho):
    """
    Load datasets for iteration 02.

    Args:
        pre (str): Prefix for the file names.
        caminho (str): Base path to the files.

    Returns:
        tuple: Tuple containing training and test data.
    """
    #print(ite, caminho)
    # Create inputs for the classifier
    X_text_tr, X_train, y_train, class_names, tipo_train, ite_train = createInputs_ite(caminho, f'{pre}Treino.csv', delimiter, 'Treino')
    X_text_ts, X_test, y_test, class_names, tipo_test, ite_test = createInputs_ite(caminho, f'{pre}Teste.csv', delimiter, 'Teste')
    return X_train, X_test    
    
def saveList(path, fileName, lista):
    """
    Save a list to a file.

    Args:
        path (str): File path.
        fileName (str): File name.
        lista (list): List to save.

    Returns:
        None
    """
    with open(path + fileName, 'w', encoding='UTF-8') as f:
        # Iterate through the list and write each item to a new line in the file
        for item in lista:
            f.write(str(item) + '\n')
    f.close
    
def criarNovoArquivo(arquivo01, arquivo02, novoArquivo):
    """
    Create a new file by concatenating two existing files.

    Args:
        arquivo01 (str): Path to the first file.
        arquivo02 (str): Path to the second file.
        novoArquivo (str): Path to the new file.

    Returns:
        None
    """
    # Open the destination file for writing
    with open(novoArquivo, 'w') as destino:
        # Open the source file A for reading
        with open(arquivo01, 'r') as origem_a:
            # Read the content of source file A and write it to the destination
            conteudo_a = origem_a.read()
            destino.write(conteudo_a + '\n')
    
        # Open the source file B for reading
        with open(arquivo02, 'r') as origem_b:
            # Read the content of source file B and write it to the destination
            conteudo_b = origem_b.read()
            destino.write(conteudo_b)
            
def concatDatasets_ite(caminho, pref):
    """
    Concatenate training and test datasets.

    Args:
        caminho (str): Base path to the files.
        pref (str): Prefix for the file names.

    Returns:
        tuple: Tuple containing concatenated data.
    """
    # Create inputs for the classifier
    X_text_tr, X_train, y_train, class_names, a, bytearray = createInputs_ite(caminho, f'{pref}Treino.csv', delimiter, 'Treino')
    X_text_ts, X_test, y_test, class_names, c, d = createInputs_ite(caminho, f'{pref}Teste.csv', delimiter, 'Teste')
    X_todos = X_train + X_test
    y_todos = y_train + y_test
    return X_todos, y_todos    
            
class DadosErros:
    def __init__(self, file, acao):
        """
        Initialize the DadosErros class.

        Args:
            file (str): Path to the file.
            acao (str): Action type.

        Returns:
            None
        """
        self.df = loadText('', file, sep=',')
        self.acao = acao

    def get_(self):
        """
        Get error data based on the action type.

        Returns:
            tuple: Tuple containing error data.
        """
        if self.acao in ['1', '2']:
            df_er = self.df.query('acao == ' + self.acao)
        else:
            df_er = self.df
            
        i = df_er['i'].to_list()
        text_pp = df_er['text_pp'].to_list()
        class_label = df_er['class_label'].to_list()
        class_label_pred = df_er['class_label_pred'].to_list()
        tipo_erro = df_er['tipo_erro'].to_list()
        classifier = df_er['classifier'].to_list()
        acao = df_er['acao'].to_list()
        ite = df_er['ite'].to_list()

        return i, text_pp, class_label, class_label_pred, tipo_erro, classifier, acao, ite            
    
# Create path
def criarCaminho(model, dts, ite='0'):
    """
    Create the path for the model and dataset.

    Args:
        model (str): Model name.
        dts (str): Dataset name.
        ite (str, optional): Iteration number. Defaults to '0'.

    Returns:
        tuple: Tuple containing the path, prefix, and model name.
    """
    pref = f'{dts}{ite}'
    md_name = md.models[model]
    caminho = f'{p.path_dts}{md_name}/{dts}/'
    md_name = md.models[model]
    return caminho, pref, md_name

def criarPasta(pasta):
    """
    Create a directory if it does not exist.

    Args:
        pasta (str): Path of the directory to create.

    Returns:
        None
    """
    # Check if the directory already exists
    if not os.path.exists(pasta):
        # If it does not exist, create the directory
        os.makedirs(pasta)    

def reset_pasta(md_name):
    """
    Delete all files in a folder and its subfolders, except those whose names match the patterns in the exception list.

    Args:
        md_name (str): Name of the model.

    Returns:
        None
    """
    folder_path = f'{p.path_dts}{md_name}'
    exception_patterns = ['*Treino.csv', '*Teste.csv']
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for name in files:
            preserved = False
            for pattern in exception_patterns:
                if fnmatch.fnmatch(name, pattern):
                    preserved = True
                    break
            
            if not preserved:
                file_path = os.path.join(root, name)
                os.remove(file_path)

        for name in dirs:
            dir_path = os.path.join(root, name)
            # Attempt to remove the directory if it is empty
            try:
                os.rmdir(dir_path)
            except OSError as e:
                pass
