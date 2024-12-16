import warnings
warnings.filterwarnings("ignore")
from time import time

from utils import io
from utils import models as md
from utils import path as p
from utils import Method as mt
from utils import ImportantFeaturesErrors as ife
from utils import importantFeatures as fi
from utils import utils as u
import pickle

etapa_final = 6

def calcular_metricas(metrics_object):
    """
    Calculate various performance metrics from the metrics object.
    """
    FP = metrics_object.fp
    FN = metrics_object.fn
    VP = metrics_object.tp
    VN = metrics_object.tn

    # Calculate Accuracy
    acuracia = (VP + VN) / (VP + VN + FP + FN)
    
    # Calculate Precision
    precisao = VP / (VP + FP) if (VP + FP) != 0 else 0
    
    # Calculate Sensitivity
    sensibilidade = VP / (VP + FN) if (VP + FN) != 0 else 0
    
    # Calculate Specificity
    especificidade = VN / (VN + FP) if (VN + FP) != 0 else 0
    
    # Calculate F1-Score
    if (precisao + sensibilidade) != 0:
        f1_score = 2 * (precisao * sensibilidade) / (precisao + sensibilidade)
    else:
        f1_score = 0
    
    return {
        'Acurácia': round(acuracia*100, 2),
        'Precisão': round(precisao*100, 2),
        'Sensibilidade': round(sensibilidade*100, 2),
        'Especificidade': round(especificidade*100, 2),
        'F1-Score': round(f1_score*100, 2),
    }

def get_metrics():
    """
    Retrieve and calculate the average performance metrics for all models.
    """
    import numpy as np
    metricas_acumuladas = {
        'Acurácia': [],
        'Precisão': [],
        'F1-Score': [],
        'Especificidade': [],
        'Sensibilidade': [],
    }

    for clf in md.models.keys():
        md_name = md.models[clf]
        for kfold in range(1, etapa_final):
            caminho, pref, md_name = io.criarCaminho(clf, kfold)
            pth = f'{caminho}md/'
            arq_model = f'{pth}{pref}{md_name}model.plk'  
            list_model_metrics = pickle.load(open(arq_model, 'rb'))
            model = list_model_metrics[0]
            metrics_treino, metrics_teste = list_model_metrics[1]
            metricas = calcular_metricas(metrics_teste)
            # print(f'Fold {kfold}: ', metricas)
            
            for key in metricas_acumuladas.keys():
                metricas_acumuladas[key].append(metricas[key])
    
    metricas_medias = {key: values + [round(np.mean(values), 2)] for key, values in metricas_acumuladas.items()}
    for key, values in metricas_medias.items():
        s = '\\textbf{' + key + ' (\%)}'
        maior = max(values)
        for value in values:
            if value == maior:
                s += ' & \\textbf{' + str(value) + '}'
            else:
                s += ' & ' + str(value)
        s += ' \\\\'
        print(s)
    return metricas_medias

def e01testarTreinarModelos():
    """
    Train and test multiple models using k-fold cross-validation and calculate errors.
    This function iterates over a dictionary of models, trains each model using k-fold 
    cross-validation, and calculates errors for each fold. The results are printed to 
    the console and logged.
    
    Steps:
    1. Train the model.
    2. Collect model errors.
    
    The function performs the following tasks:
    - Iterates over each classifier in the dictionary of models.
    - For each classifier, performs k-fold cross-validation.
    - Trains the model and measures the training time.
    - Collects model errors and updates the error log.
    """
    for clf in md.models.keys():
        inicio = time()
        md_name = md.models[clf]
        print(f'################################################################################### {md_name.upper()} ###################################################################################')
    
        print(f'************************************************************************** Train/Test and Calculate Errors ************************************************************************')
        for kfold in range(1, etapa_final):
            print(f'------------------------------------------------------------------------------------------ KFOLD: {kfold} -------------------------------------------------------------------------------')
    
            print('Step 1: Training the model...', end="")
            inicio_etapa = time()
            caminho, pref, md_name = io.criarCaminho(clf, kfold)
    
            X_train_all, X_train, y_train, _, _, X_test_all, X_test, y_test, _, _, _ = io.loadDatasets_ite(pref, caminho)
    
            model, metrics = md.createModel(md.tfidfVectorizer, clf, X_train, y_train, X_test, y_test, -2) 
            fim_etapa = time()
            print(f'Completed in {fim_etapa - inicio_etapa:.2f} seconds')
    
            inicio_etapa = time()
            print('Step 2: Collecting model errors...', end="")
            mt.executeTrainingAndErrorAnalysis(caminho, pref, md_name, X_train, y_train, X_test, X_test_all, y_test, model)
            mt.updateErrorLog(caminho, pref, md_name)
    fim = time()
    print(f'##################################################################################### END: {u.ctp(inicio, fim)} ############################################################################')

def e02explicarModelos():
    """
    This function explains models using logistic regression as the intrinsic explainer.
    It iterates over a dictionary of models, and for each model, performs the following steps:
    1. Prints the model name.
    2. For each k-fold in the range from 1 to `etapa_final`:
        a. Creates paths and prefixes to save the results.
        b. Loads the training and test datasets.
        c. Creates a model using the TF-IDF vectorizer and the classifier.
        d. Explains all sentences in the training set using the specified explainer.
        e. Prints the time taken for each k-fold iteration.
    3. Prints the total time taken to explain the model.
    """
    for clf in md.models.keys():
        inicio = time()
        md_name = md.models[clf]
        if md_name == 'lrg':
            explainer = 'logistic_regression'  # Intrinsic explainer logistic_regression
        else:
            explainer = 'lime'  # Global explainer Lime
        
        print(f'################################################################################### {md_name.upper()} ###################################################################################')
        
        print(f'*************************************************************************** Explain Training Sentences **************************************************************************')
        for kfold in range(1, etapa_final):
            caminho, pref, md_name = io.criarCaminho(clf, kfold)
            print(f'------------------------------------------------------------------------------------------ KFOLD: {kfold} -------------------------------------------------------------------------------')
            inicio_etapa = time()
            print('Step 3: Explaining the model...', end="")
            inicio_etapa = time()
            caminho, pref, md_name = io.criarCaminho(clf, kfold)
            X_train, y_train, X_test, y_test = io.loadDatasets_ite01(pref, caminho)
            model, metrics = md.createModel(md.tfidfVectorizer, clf, X_train, y_train, X_test, y_test, -1)
            vectorizer = model.steps[0][1]
            model_explaner = model.steps[1][1]
            sentencas = X_train
            mt.explicarTodas(caminho, model, pref, md_name, sentencas, explainer=explainer, vectorizer=vectorizer, model_explaner=model_explaner)
            fim_etapa = time()
            print(f'Completed in {fim_etapa - inicio_etapa:.2f} seconds')
            
        fim = time()
        print(f'##################################################################################### END: {u.ctp(inicio, fim)} ############################################################################')

def e03calcularFI():
    """
    Calculate and print the feature importance for different models and k-folds.
    This function iterates over the models defined in the `md.models` dictionary.
    For each model, it performs the following steps:
    1. For each k-fold (from 1 to `etapa_final`):
        a. Creates the path and prefix for the current model and k-fold using `io.criarCaminho`.
        b. Prints the current k-fold number.
        c. Calls `mt.criarIF` to create the important features for the model.
        d. Reads the important features using `fi.lerIf`.
        e. Prints the time taken for the current k-fold.
    2. Prints the total time taken for the current model.
    """
    for clf in md.models.keys():
        inicio = time()
        md_name = md.models[clf]
        print(f'################################################################################### {md_name.upper()} ###################################################################################')
        
        print(f'****************************************************************** Calculate Important Features ******************************************************************')
        for kfold in range(1, etapa_final):
            caminho, pref, md_name = io.criarCaminho(clf, kfold)
            print(f'------------------------------------------------------------------------------------------ KFOLD: {kfold} -------------------------------------------------------------------------------')
            inicio_etapa = time()
            print('Step 4: Creating important features for the model...', end="")
            mt.criarIF(caminho, pref, md_name)
            fi_global = fi.lerIf(caminho, pref, md_name, 'Lime')
            fim_etapa = time()
            print(f'Completed in {u.ctp(inicio_etapa, fim_etapa)}')
        
        fim = time()
        print(f'##################################################################################### END: {u.ctp(inicio, fim)} ############################################################################')


def e04calcularConf():
    """
    Executes the process of calculating and recording error characteristics for different classifiers and k-folds.
    This function iterates over classifiers and k-folds, performing the following steps for each combination:
    1. Creates paths and prefixes to store the results.
    2. Retrieves similar sentences for 'other' and 'same' types.
    3. Counts occurrences of similar sentences.
    4. Creates CSV files with frequency data of error characteristics.
    5. Optimizes and counts occurrences for test data.
    The function prints progress and time information for each step.
    """
    for clf in md.models.keys():
        start = time()
        md_name = md.models[clf]
        print(f"################################################################################### {md_name.upper()} ###################################################################################")

        for kfold in range(1, etapa_final):
            caminho, pref, md_name = io.criarCaminho(clf, kfold)
            print(f"------------------------------------------------------------------------------------------ KFOLD: {kfold} -------------------------------------------------------------------------------")
            print("Step 4: Selecting similar sentences in the training set...")
            start_step = time()

            # Call to the function to retrieve similar sentences - 'other'
            print("Starting retrieval of similar sentences 'other'...")
            start_function = time()
            similar_by_error_type = mt.recuperarSentencasSemelhantesTreinoPorClasse(caminho, pref, md_name, outro=True)
            end_function = time()
            print(f"Retrieval of similar sentences 'other' completed in {u.ctp(start_function, end_function)}")
            io.gravarJson(f'{caminho}{p.path_if}{pref}{md_name}Semelhantes0Outro.json', similar_by_error_type[0])
            io.gravarJson(f'{caminho}{p.path_if}{pref}{md_name}Semelhantes1Outro.json', similar_by_error_type[1])

            # Call to the function to retrieve similar sentences - 'same'
            print("Starting retrieval of similar sentences 'same'...")
            start_function = time()
            similar_by_error_type_same = mt.recuperarSentencasSemelhantesTreinoPorClasse(caminho, pref, md_name, outro=False)
            end_function = time()
            print(f"Retrieval of similar sentences 'same' completed in {u.ctp(start_function, end_function)}")
            io.gravarJson(f'{caminho}{p.path_if}{pref}{md_name}Semelhantes0Mesmo.json', similar_by_error_type_same[0])
            io.gravarJson(f'{caminho}{p.path_if}{pref}{md_name}Semelhantes1Mesmo.json', similar_by_error_type_same[1])

            print('Counting occurrences')
            start_count = time()
            other = ife.contar_ocorrencias_semelantes(caminho, pref, md_name, 'outro')
            same = ife.contar_ocorrencias_semelantes(caminho, pref, md_name, 'mesmo')

            zero = 0
            one = 1
            csv_path = f'{caminho}{p.path_if}{pref}{md_name}ErrorFeatures{one}.csv'
            ife.create_frequency_csv(other[1][zero], same[0][zero], other[0][zero], csv_path)

            csv_path = f'{caminho}{p.path_if}{pref}{md_name}ErrorFeatures{zero}.csv'
            ife.create_frequency_csv(other[1][one], same[0][one], other[0][one], csv_path)
            end_count = time()
            print(f'End of counting occurrences in {u.ctp(start_count, end_count)}')

            csv_path = f'{caminho}{p.path_if}{pref}{md_name}ErrorFeatures{zero}.csv'
            ife.create_frequency_csv(other[1][one], same[0][one], other[0][one], csv_path)
            ife.contar_ocorrencias_semelantes_teste_mesmo_otimizado(caminho, pref, md_name)
            end_step = time()
            print(f'Finished in {u.ctp(start_step, end_step)}')

            end_step = time()
            print(f'Finished in {u.ctp(start_step, end_step)}')

        end = time()
        print(f'Total execution time for classifier {md_name}: {u.ctp(start, end)}')


def e05gerarPadroesInfluencias():
    """
    Generate influence patterns for each classifier model in the `md.models` dictionary.
    This function iterates over each classifier model, calculates the influence patterns for each fold,
    and prints the time taken for each step and for the overall process.
    
    Steps:
    1. Iterate over each classifier model in `md.models`.
    2. For each classifier model:
        a. Print the model name.
        b. Calculate the influence patterns for each fold from 1 to `etapa_final`.
        c. Print the time taken to calculate the influence patterns.
        d. Print the total time taken for the entire process.
    """
    for clf in md.models.keys():
        inicio = time()
        md_name = md.models[clf]
        print(f'################################################################################### {md_name.upper()} ###################################################################################')
        
        print(f'************************************************************************** Calculate Spurious Influence Patterns **************************************************************')
        inicio_etapa = time()

        for kfold in range(1, etapa_final):
            mt.calcula_padroes(clf, str(kfold))
            
        fim_etapa = time()
        print(f'Completed in {u.ctp(inicio_etapa, fim_etapa)}')
        fim = time()
        print(f'##################################################################################### END: {u.ctp(inicio, fim)} ############################################################################')

def e06gerarResultados():
    """
    Generate results for each classifier model.
    This function iterates over all classifier models defined in `md.models`, performs various steps to process and 
    summarize data, and calculates the final results. The steps include:
    1. Gather data from the K-folds.
    2. Summarize data.
    3. Calculate results.
    The duration of each step is printed to the console.
    
    Steps:
    - Gather data from the K-folds.
    - Summarize data.
    - Calculate results.
    """
    for clf in md.models.keys():
        inicio = time()
        md_name = md.models[clf]
        print(f'################################################################################### {md_name.upper()} ###################################################################################')
    
        print(f'************************************************************************* Gather all data and calculate the result ************************************************************')
        print('Step 9: Gather data from the K-folds', end="")
        inicio_etapa = time()
        mt.juntarDados(clf)
        mt.juntarDados(clf, 0, 'if')
        mt.juntarDados(clf, 1, 'if')
        mt.juntarDados(clf, 0, 'ef')
        mt.juntarDados(clf, 1, 'ef')
        
        fim_etapa = time()
        print(f'Completed in {u.ctp(inicio_etapa, fim_etapa)}')

        print('Step 10: Summarize data', end="")
        inicio_etapa = time()
        caminho, pref, md_name = io.criarCaminho(clf, 6)
        mt.sumarizar_ce(caminho, pref, md_name)
        mt.sumarizar_if(caminho, pref, md_name)
        mt.sumarizar_ef(caminho, pref, md_name)
        mt.mover_arquivos(caminho, pref, md_name)
        fim_etapa = time()
        print(f'Completed in {u.ctp(inicio_etapa, fim_etapa)}')

        print('Step 11: Calculate result', end="")
        inicio_etapa = time()
        mt.criarImportanciaErros(f'{caminho}{p.path_if}', pref, md_name, 0)
        mt.criarImportanciaErros(f'{caminho}{p.path_if}', pref, md_name, 1)
        mt.criarArquivoComparacao(caminho, pref, md_name, '0')
        mt.criarArquivoComparacao(caminho, pref, md_name, '1')
        fim_etapa = time()
        print(f'Completed in {u.ctp(inicio_etapa, fim_etapa)}')

        fim = time()
        print(f'##################################################################################### END: {u.ctp(inicio, fim)} ############################################################################')


if __name__ == '__main__':
    # Start of the process
    inicio00 = time()
    print(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ START @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    e01testarTreinarModelos()
    e02explicarModelos()
    e03calcularFI()
    e04calcularConf()
    e05gerarPadroesInfluencias()
    e06gerarResultados()
    get_metrics()
    fim00 = time()

    print(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ END: {u.ctp(inicio00, fim00)} @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
