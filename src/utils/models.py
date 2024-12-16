# Importing packages

import nltk

from nltk.stem.porter import *
from utils import io as io
from utils import preProcessamento as pp

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression, SGDClassifier
from nltk.stem.porter import *
from sklearn.pipeline import make_pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


def tokenize(text):
    """
    Tokenize the text, removing stopwords.

    Args:
        text (str): Text to tokenize.

    Returns:
        list: List of tokenized words.
    """
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        if (item not in stop_words) and (item not in stop_words_domain): 
            stems.append(item)
    return stems


def extractNamesModel(model):
    """
    Extract the names of the feature extractor and classifier from the model.

    Args:
        model (Pipeline): Scikit-learn pipeline model.

    Returns:
        tuple: Feature extractor name and classifier name.
    """
    featureExtractor = featureExtractor_names[str(model.steps[0][1])[0:str(model.steps[0][1]).index('(')]]
    classifier = models_names[str(model.steps[1][1])[0:str(model.steps[1][1]).index('(')]]

    return featureExtractor, classifier

class Metrics:
    def __init__(self, y_pred, y, tipo):
        """
        Initialize the Metrics class with evaluation metrics.

        Args:
            y_pred (list): Predicted labels.
            y (list): True labels.
            tipo (str): Type of data (e.g., 'Train', 'Test').
        """
        dec = 3
        # Confusion matrix
        conf_matrix = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = conf_matrix.ravel()

        # Accuracy
        acc = round(accuracy_score(y, y_pred), dec)

        # Precision
        pre = round(precision_score(y, y_pred), dec)

        # Recall (sensitivity)
        sen = round(recall_score(y, y_pred), dec)

        # Specificity
        esp = round(tn / (tn + fp), dec)

        # F1-score
        f1 = round(f1_score(y, y_pred), dec)

        self.tipo = tipo
        self.f1 = f1
        self.tn = tn
        self.fp = fp
        self.fn = fn
        self.tp = tp
        self.acc = acc
        self.pre = pre
        self.sen = sen
        self.esp = esp
        self.mc = conf_matrix
        self.y_o = y
        self.y_p = y_pred

    def diff(self, metrica_parametro, nome_da_propriedade):
        """
        Calculate the difference between two metrics.

        Args:
            metrica_parametro (Metrics): Metrics object to compare with.
            nome_da_propriedade (str): Name of the metric property.

        Returns:
            float: Difference between the metrics.
        """
        valor_self = getattr(self, nome_da_propriedade, None)
        valor_parametro = getattr(metrica_parametro, nome_da_propriedade, None)
        if valor_self is not None and valor_parametro is not None:
            return round(valor_self - valor_parametro, 4)

    @staticmethod
    def formatar_saida(mf, mi):
        """
        Format the output of the metrics comparison.

        Args:
            mf (Metrics): Final metrics.
            mi (Metrics): Initial metrics.

        Returns:
            str: Formatted string of the metrics comparison.
        """
        # Define the metrics to be compared
        metricas = ['acc', 'pre', 'f1', 'esp', 'tn', 'fn', 'sen', 'tp', 'fp']
        saida = []

        # Iterate over each metric, obtaining the values from mf and mi, and calculate the difference
        for metrica in metricas:
            valor_mi = getattr(mi, metrica)
            valor_mf = getattr(mf, metrica)
            diferenca = mf.diff(mi, metrica)

            # Add the values to the output list
            saida.extend([valor_mi, valor_mf, round(diferenca, 3)])

        # Convert the list to a formatted string
        saida_formatada = ','.join(map(str, saida))
        return saida_formatada
    
    def print_metrics(self):
        """
        Print the evaluation metrics.

        Returns:
            None
        """
        print(self.tipo, "Ac:", self.acc, "Pr:", self.pre, "Re:", self.sen, 'F1', self.f1, "Esp:", self.esp)
        
        print('Confusion Matrix:')
        print('       Predicted N   Predicted P')
        print('Real N [' + str(self.tn) + '       ' + str(self.fp) + ']')
        print('Real P [' + str(self.fn) + '       ' + str(self.tp) + ']')

def trainModel(featureExtractor, classifier, X_train, y_train, print_):
    """
    Train a model with the given feature extractor and classifier.

    Args:
        featureExtractor (object): Feature extraction method.
        classifier (object): Classifier method.
        X_train (list): Training data.
        y_train (list): Training labels.
        print_ (bool): Whether to print the training metrics.

    Returns:
        Metrics: Training metrics.
    """
    X_train = pp.preProcessaTexto03(X_train)
    c = make_pipeline(featureExtractor, classifier)
    c.fit(X_train, y_train)
    featureExtractor, classifier = extractNamesModel(c)
    metrics_treino = Metrics(c.predict(X_train), y_train, 'Train:')

    if print_:
        print(featureExtractor + ' + ' + classifier)
        metrics_treino.print_metrics()
        
    return metrics_treino


def predict(model, X, y):
    """
    Predict labels for the given data using the trained model.

    Args:
        model (object): Trained model.
        X (list): Data to predict.
        y (list): True labels.

    Returns:
        list: Predicted labels.
    """
    X1 = X.copy()
    X1 = pp.preProcessaTexto03(X1)
    y_retorno = model.predict(X1)
        
    return y_retorno
    
# ALTERAR
def predictModel(model, X, y, tipo, print_ = False, ppp = False):
    """
    Predict labels and calculate metrics for the given data using the trained model.

    Args:
        model (object): Trained model.
        X (list): Data to predict.
        y (list): True labels.
        tipo (str): Type of data (e.g., 'Train', 'Test').
        print_ (bool, optional): Whether to print the metrics. Defaults to False.
        ppp (bool, optional): Whether to preprocess the data. Defaults to False.

    Returns:
        Metrics: Calculated metrics.
    """
    X1 = X.copy()
    if ppp:
        X1 = pp.preProcessaTexto03(X1)

    metrics = Metrics(model.predict(X1), y, f'{tipo}: ')

    if print_:
        metrics.print_metrics()
        
    return metrics
    
def createModel(featureExtractor, classifier, X_train, y_train, X_test, y_test, print_ = -1):
    """
    Create and train a model, then calculate metrics for training and test data.

    Args:
        featureExtractor (object): Feature extraction method.
        classifier (object): Classifier method.
        X_train (list): Training data.
        y_train (list): Training labels.
        X_test (list): Test data.
        y_test (list): Test labels.
        print_ (int, optional): Whether to print the metrics. Defaults to -1.

    Returns:
        tuple: Trained model and a list of training and test metrics.
    """
    X_train = pp.preProcessaTexto03(X_train)
    X_test = pp.preProcessaTexto03(X_test)
    c = make_pipeline(featureExtractor, classifier)
    c.fit(X_train, y_train)
    featureExtractor, classifier = extractNamesModel(c)
    metrics_treino = Metrics(c.predict(X_train), y_train, 'Train: ')
    metricsTeste = Metrics(c.predict(X_test), y_test, 'Test: ')

    if print_ == 0:
        print(featureExtractor + ' + ' + classifier)
        metrics_treino.print_metrics()
    elif print_ == 1: 
        metricsTeste.print_metrics()
    elif print_ == 2: 
        metrics_treino.print_metrics()
        metricsTeste.print_metrics()        
        
    return c, [metrics_treino, metricsTeste]

# Identification and Grouping of Classification Errors
def getErrorsModel(model, X, y, tipo, pref, caminho):
    """
    Identify and group classification errors.

    Args:
        model (tuple): Tuple containing the trained model and its name.
        X (list): Data to predict.
        y (list): True labels.
        tipo (str): Type of data (e.g., 'Train', 'Test').
        pref (str): Prefix for the file names.
        caminho (str): Base path to the files.

    Returns:
        list: List of tuples containing error details.
    """
    y_preds = model[0].predict(X)
    
    text_pp, class_label, class_label_pred, tipo_erro, classifier, i_, acao, ite, tipo_ = ([] for _ in range(9))

    for i in range(len(X)):
        if y[i] != y_preds[i]:
            text_pp.append(X[i])
            class_label.append(y[i])
            class_label_pred.append(y_preds[i])
            classifier.append(model[1])
            i_.append(i + 2)
            acao.append('0')
            ite.append(pref)
            tipo_.append(tipo)
            if y_preds[i] == 1:
                tipo_erro.append('FP')
            else:
                tipo_erro.append('FN')
                
    return list(zip(i_, text_pp, class_label, class_label_pred, tipo_erro, classifier, acao, ite, tipo_))


# Definition of work variables
nltk_stopwords = nltk.corpus.stopwords.words('portuguese')
stop_words_domain = ['pi', 'p', 'sao', 'doze', 'quant', 'nazaria', 'brandao', 'brasileira', 'vez', 'semestre2018', 'santos', 'piaui', 'pmpii']
stop_words = set(stopwords.words('portuguese')) 

n_feat = 10000
tfidfVectorizer = TfidfVectorizer(tokenizer=tokenize, ngram_range=(1, 1), max_features=n_feat)

mapExtractorFeature = dict()
mapExtractorFeature['TfidfVectorizer'] = tfidfVectorizer

dec = 3

# Classifiers
# Configuring the SGDClassifier for linear SVM with squared hinge loss
sgdDClassifier         = SGDClassifier(loss='squared_hinge', random_state=42)
logisticRegression     = LogisticRegression(C=5,random_state=42)

# Dictionary of classifiers to be used
models = {
            sgdDClassifier: 'Sgd',
            logisticRegression: 'Lrg',
        }
        
models_names = {
                'SGDClassifier': 'Sgd',
                'LogisticRegression':'Lrg'
                }    

models_names_ef = {
                    'SGDClassifierTfidfVectorizer': 'SgdTfIdf',
                    'LogisticRegressionTfidfVectorizer': 'LrgTfIdf'
                    }    

featureExtractor_names = {'TfidfVectorizer': 'TfIdf'}

def loadModel(caminho, pref, md_name):
    """
    Load the model from a file.

    Args:
        caminho (str): Base path to the files.
        pref (str): Prefix for the file names.
        md_name (str): Name of the model.

    Returns:
        tuple: Loaded model and its initial metrics.
    """
    # Load the model from the file
    with open(f'{caminho}md/{pref}{md_name}model.plk', 'rb') as arquivo:
        modelo_carregado = pickle.load(arquivo)

    mo, mi = modelo_carregado
    return mo, mi
