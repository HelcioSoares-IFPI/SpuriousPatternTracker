import warnings
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from operator import itemgetter
import os
import shutil
warnings.filterwarnings("ignore")

from logistic_regression import get_score

from utils import io
from utils import LimeUtils as u
from utils import models as md
from utils import path as p
from utils import LimeUtils as u
from utils import ImportantFeaturesErrors as ife
from utils import importantFeatures as fi
from utils import preProcessamento as pp

n = 15

#### EXPLAIN ALL SENTENCES
# Separate Words by Class Based on Explanation Weights
# Process JSON Data
def process_json_data(filepath):
    """
    Process JSON data to separate words by class based on explanation weights.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        tuple: Two lists containing words and their values for class 0 and class 1.
    """
    # Reading the JSON file
    with open(filepath, 'r') as file:
        data = json.load(file)

    # Initializing lists for classes 0 and 1
    class_0_list = []
    class_1_list = []

    # Processing each key in the JSON dictionary
    for key in data:
        for index, class_data in enumerate(data[key]):
            # Iterating over each word in the class
            for word, value in class_data.items():
                # Adding the word and value to the appropriate list
                if index == 0:
                    if value >= 0:
                        class_0_list.append([word, value])
                    else:
                        class_1_list.append([word, value])
                else:
                    if value >= 0:
                        class_1_list.append([word, value])
                    else:
                        class_0_list.append([word, value])

    return class_0_list, class_1_list

# Generate and Save Local Explanations for All Sentences
def explicarTodas(caminho, model, pref, md_name, sentencas, explainer='lime', **kwargs):
    """
    Generate and save local explanations for all sentences.
    
    Parameters:
    - caminho: Path to save the explanations.
    - model: The model to be explained.
    - pref: Prefix for the saved file name.
    - md_name: Model name to be included in the file name.
    - sentencas: List of sentences to be explained.
    - explainer: Type of explainer to use ('lime' or 'logistic_regression').
    - kwargs: Additional arguments for the logistic regression explainer.
    """
    todas_explicacoes = dict()
    for i in tqdm(range(len(sentencas))):
        text_to_explain = sentencas[i]
        if len(text_to_explain) > 0:
            if explainer == 'lime':
                fl = u.localExplain_v(u.explainer, model, text_to_explain, False)
            elif explainer == 'logistic_regression':
                modelExplainer = kwargs['model_explaner']
                vectorizer = kwargs['vectorizer']
                fl = get_score(vectorizer, modelExplainer, text_to_explain)
            todas_explicacoes[text_to_explain] = fl
    io.gravarJson(f'{caminho}{pref}{md_name}allLocalExp_tr_ts.json', todas_explicacoes)


# Create important features file
def criarIF(caminho, pref, md_name):
    """
    Create a file of important features.

    Args:
        caminho (str): Base path to the files.
        pref (str): Prefix for the file names.
        md_name (str): Name of the model.

    Returns:
        None
    """
    file_path = f'{caminho}{pref}{md_name}allLocalExp_tr_ts.json'
    class_0, class_1 = process_json_data(file_path)

    caminho_if = f'{caminho}{p.path_if}'
    io.criarPasta(caminho_if)

    # Save the list to a JSON file
    with open(f'{caminho_if}{pref}{md_name}ListClass0.json', 'w', encoding='utf-8') as arquivo:
        json.dump(class_0, arquivo, ensure_ascii=False, indent=4)

    # Save the list to a JSON file
    with open(f'{caminho_if}{pref}{md_name}ListClass1.json', 'w', encoding='utf-8') as arquivo:
        json.dump(class_1, arquivo, ensure_ascii=False, indent=4)

# Execute Training and Test with Error Analysis
def executeTrainingAndErrorAnalysis(caminho, pref, md_name, X_train, y_train, X_test, y_test, model):
    """
    Execute training and test with error analysis.

    Args:
        caminho (str): Base path to the files.
        pref (str): Prefix for the file names.
        md_name (str): Name of the model.
        X_train (list): Training data.
        y_train (list): Training labels.
        X_test (list): Test data.
        y_test (list): Test labels.
        model (object): Trained model.

    Returns:
        None
    """
    data = []
    data = data + md.getErrorsModel([model, md_name], X_test, y_test, md.TESTE, pref, caminho)
    df = pd.DataFrame(data, columns=['i', 'text_pp', 'class_label', 'class_label_pred', 'tipo_erro', 'classifier', 'acao', 'ite', 'tipo'])
    nome_arquivo_erro = f'{caminho}/{pref}{md_name}Erros.csv'
    df.to_csv(nome_arquivo_erro, sep=',', index=False)

def updateErrorLog(caminho, pref, md_name):
    """
    Update the error log with new errors.

    Args:
        caminho (str): Base path to the files.
        pref (str): Prefix for the file names.
        md_name (str): Name of the model.

    Returns:
        None
    """
    nome_arquivo = f'{caminho}{pref}{md_name}Erros.csv'
    nome_arquivo_teste = f'{caminho}{pref}Teste.csv'
    df_ts = pd.read_csv(nome_arquivo_teste)
    text_ts = df_ts['text'].to_list()

    # Load the treated error DataFrame
    df_td_er = pd.read_csv(f'./{md_name}Erros.csv')
    text_td_er = df_td_er['text_pp'].to_list()
    y_td_er = df_td_er['class_label'].to_list()
    text_er01 = df_td_er['text'].to_list()

    # Adjust the formatting of y_td_er values
    for i in range(len(y_td_er)):
        y_td_er[i] = str(y_td_er[i]).replace('.0', '')

    # Load the new error DataFrame for updating
    df_erros = pd.read_csv(nome_arquivo)
    i_er = df_erros['i'].to_list()
    text_pp_e = df_erros['text_pp'].to_list()
    y_erros = df_erros['class_label'].to_list()

    # Update text_td_er and y_td_er with new values or modify as needed
    i0 = 0
    for i, text_er in enumerate(text_pp_e):
        if text_er not in text_td_er:
            text_td_er.append(text_er)
            # This is where we assign the corresponding y_erros value to text_er
            y_td_er.append(y_erros[i])
            text_er01.append(text_ts[i_er[i0] - 2])
        else:
            # If the text already exists in text_td_er, you can choose to update it
            # with the corresponding y_erros value if necessary. For example:
            # index = text_td_er.index(text_er)
            # y_td_er[index] = y_erros[i]
            pass
        i0 = i0 + 1
    # Creating a new DataFrame and saving the updated results
    data = list(zip(text_er01, y_td_er, text_td_er))
    df = pd.DataFrame(data, columns=['text', 'class_label', 'text_pp'])
    df.to_csv(f'./{md_name}Erros.csv', sep=',', index=False)

### RECOVER MODEL ERRORS
# Recover Similar Sentences by Class
def recuperarSentencasSemelhantesTreinoPorClasse(caminho, pref, md_name, outro=True):
    """
    Recover similar training sentences by class.

    Args:
        caminho (str): Base path to the files.
        pref (str): Prefix for the file names.
        md_name (str): Name of the model.
        outro (bool): Whether to reverse the class order. Defaults to True.

    Returns:
        list: Two dictionaries containing similar sentences for each class.
    """
    df_tr = pd.read_csv(f'{caminho}{pref}Treino.csv')
    if outro:
        # Create a list with training sentences of class 0 and class 1, in reverse order
        lft = [(df_tr[df_tr['class_label'] == 1])['text_pp'].to_list(), (df_tr[df_tr['class_label'] == 0])['text_pp'].to_list()]
        df_e = pd.read_csv(f'{caminho}{pref}{md_name}Erros.csv')
        lfe = [(df_e[df_e['class_label'] == 0])['text_pp'].to_list(), (df_e[df_e['class_label'] == 1])['text_pp'].to_list()]
    else:
        # Create a list with training sentences of class 0 and class 1, in reverse order
        lft = [(df_tr[df_tr['class_label'] == 0])['text_pp'].to_list(), (df_tr[df_tr['class_label'] == 1])['text_pp'].to_list()]
        df_e = pd.read_csv(f'{caminho}{pref}{md_name}Erros.csv')
        lfe = [(df_e[df_e['class_label'] == 0])['text_pp'].to_list(), (df_e[df_e['class_label'] == 1])['text_pp'].to_list()]

    semelhantes = [dict(), dict()]

    for ii in range(2):
        se_list = lfe[ii]
        st_list = lft[ii]
        for s_ in se_list:
            st_list_ = st_list + [s_]

            vectorizer = md.tfidfVectorizer
            tfidf_matrix = vectorizer.fit_transform(st_list_)
            cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
            most_similar_indices = np.argsort(-cosine_similarities)
            
            threshold = 0.1  # Initialize similarity threshold with 0.1
            decrement = 0.01  # Set the decrement value for the threshold
            found = False  # Initialize the found variable as False
            
            # Loop that decrements the threshold value until similar sentences are found or the threshold is less than or equal to 0
            while threshold > 0 and not found:
                most_similar_sentences = [(st_list[i], cosine_similarities[i]) for i in most_similar_indices if cosine_similarities[i] >= threshold]
                
                if most_similar_sentences:  # Check if any sentence was found
                    found = True  # Update the found variable to True, indicating that similar sentences were found
                else:
                    threshold -= decrement  # Decrement the threshold value
            
            # If no similar sentence was found after the loop, assign '_NONE_' to the list
            if not found:
                most_similar_sentences = [('_NONE_', 0)]
            
            semelhantes[ii][s_] = most_similar_sentences
    return semelhantes

# Recover Influential Words in Errors
def recuperaPalavrasInfluentesErro(semelhantes_por_tipo_erro):
    """
    Recover influential words for errors based on similar sentences.

    Args:
        semelhantes_por_tipo_erro (list): List of dictionaries containing similar sentences for each error type.

    Returns:
        list: List of dictionaries with main sentences and their similar sentences.
    """
    ms = []
    for semelhantes in semelhantes_por_tipo_erro:
        
        main_sentences = dict()
        for a, most_similar_sentences in semelhantes.items():
            sentence_lists = []
            for sentence, y_t in most_similar_sentences:
                sentence_lists.append(sentence)
            main_sentences[a] = sentence_lists   
        ms.append(main_sentences)
    return ms


# Filter Common Keys by Maximum Values
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

def somar_perturbacao(a, b):
    """
    Sum perturbation values, considering negative values.

    Args:
        a (float): First value.
        b (float): Second value.

    Returns:
        float: Sum of the perturbation values, rounded to three decimal places.
    """
    # Check if both numbers are negative
    if a < 0 and b < 0:
        # If both are negative, sum the absolute values and make the result negative
        resultado = -(abs(a) + abs(b))
    else:
        # For other cases (at least one positive or zero), sum the absolute values
        resultado = abs(a) + abs(b)

    # Return the result rounded to three decimal places
    return round(resultado, 3)
    
def somar_acertos(a, b):
    """
    Sum accuracy values.

    Args:
        a (float): First value.
        b (float): Second value.

    Returns:
        float: Sum of the accuracy values, rounded to three decimal places.
    """
    resultado = a + b
    return round(resultado, 3)
    
def calc_cat(cl, dif_esp, dif_sen):
    """
    Calculate the category based on specificity and sensitivity differences.

    Args:
        cl (int): Class label (0 or 1).
        dif_esp (float): Difference in specificity.
        dif_sen (float): Difference in sensitivity.

    Returns:
        int: Category code.
    """
    if dif_esp < 0 and dif_sen < 0:
        return 0  # Important word for both classes

    if cl == 0:
        if dif_esp < 0 and dif_sen > 0:
            return 4  # Class 0 words 1, decreases specificity and increases sensitivity. Not spurious correlation.
        elif dif_esp > 0 and dif_sen < 0:
            return 1  # Class 0 words 1, increases specificity and decreases sensitivity. Could be spurious correlation.
        elif dif_esp > 0 and dif_sen == 0:
            return 2  # Class 0 words 1, increases specificity and does not change sensitivity. Could be spurious correlation.
    elif cl == 1:
        if dif_sen < 0 and dif_esp > 0:
            return 5  # Class 1 words 0, increases specificity and decreases sensitivity. Not spurious correlation.
        elif dif_sen > 0 and dif_esp < 0:
            return 1  # Class 1 words 0, increases sensitivity and decreases specificity. Could be spurious correlation.
        elif dif_sen > 0 and dif_esp == 0:
            return 2  # Class 1 words 0, increases sensitivity and does not change specificity. Could be spurious correlation.

    if dif_esp > 0 and dif_sen > 0:
        return 3  # Increases both metrics

    return 4  # Others

# Remove Words from Sentences
def retirar_palavras_das_sentencas(word, sentence_list):
    """
    Remove specific words from a list of sentences.

    Args:
        word (str): Words to remove, separated by spaces.
        sentence_list (list): List of sentences.

    Returns:
        list: List of sentences with the specified words removed.
    """
    set_a = set(word.split())  # Convert the 'word' expression into a set for quick verification
    result = []  # Initialize the result list
    
    for sentence in sentence_list:
        list_b = sentence.split()  # Split the current sentence into a list of words to preserve order
        
        # Check if all words in 'set_a' are in 'list_b'
        if set_a.issubset(set(list_b)):
            # Construct a new list without the words from 'set_a'
            result_list = [word for word in list_b if word not in set_a]
            # Add the modified sentence to the result list
            result.append(' '.join(result_list))
        else:
            # If not all words from 'set_a' are in 'list_b', add the original sentence
            result.append(sentence)
    
    return result
    
# Calculate metric losses for each analyzed pattern
def printCorrelacao01(padrao, cl, pref, tipo, mi, model, X_test, y_test):
    """
    Calculate correlation metrics for a given pattern.

    Args:
        padrao (str): Pattern to analyze.
        cl (int): Class label.
        pref (str): Prefix for the file names.
        tipo (str): Type of data (e.g., 'Test').
        mi (list): Initial model metrics.
        model (object): Trained model.
        X_test (list): Test data.
        y_test (list): Test labels.

    Returns:
        list: List of formatted correlation metrics.
    """
    cor = []
    X_test = retirar_palavras_das_sentencas(padrao, X_test)
    #print(f'---------------------- {padrao} ----------------------')  
    # Only the test base will be analyzed
    X = X_test.copy()

    mf = [md.predictModel(model, X, y_test, tipo, False, False)]

    # Store properties in temporary variables to avoid multiple accesses
    mf_esp, mf_sen, mf_tp, mf_tn, mf_fn, mf_fp = mf[0].esp, mf[0].sen, mf[0].tp, mf[0].tn, mf[0].fn, mf[0].fp
    mi_esp, mi_sen, mi_tp, mi_tn, mi_fn, mi_fp = mi[1].esp, mi[1].sen, mi[1].tp, mi[1].tn, mi[1].fn, mi[1].fp

    dif_esp = round(mf_esp - mi_esp, 3)
    dif_sen = round(mf_sen - mi_sen, 3)
    dif_tp = round(mf_tp - mi_tp, 3)
    dif_tn = round(mf_tn - mi_tn, 3)
    dif_fn = round(mf_fn - mi_fn, 3)
    dif_fp = round(mf_fp - mi_fp, 3)

    corEspuria = dif_sen > 0 or dif_esp > 0

    pQtd = somar_perturbacao(dif_fp, dif_fn)  # \varrho_p Number of perturbed items (that change prediction after base perturbation)
    aQtd = somar_acertos(dif_tp, dif_tn)      # \varrho_a Number of correct predictions after perturbation

    pIdx = somar_perturbacao(dif_sen, dif_esp) # \tau_p: Perturbation rate
    aIdx = somar_acertos(dif_esp, dif_sen)    # \tau_a: Accuracy rate

    saida_formatada = md.Metrics.formatar_saida(mf[0], mi[1])
    cat = calc_cat(cl, dif_esp, dif_sen)
    cor.append(f'{tipo},{pref},{cl},{saida_formatada},{pIdx},{pQtd},{aIdx},{aQtd},{cat},{corEspuria},{padrao}')
    return cor

### CALCULATE SPURIOUS PATTERNS. NOTE: CHECK THE NEW NOMENCLATURE
# Convert List to CSV
def lista_para_csv(lista, nome_arquivo_csv):
    """
    Convert a list to a CSV file.

    Args:
        lista (list): List of strings, where the first element contains the headers and the rest contain the data.
        nome_arquivo_csv (str): Name of the CSV file to save.

    Returns:
        None
    """
    # The first line of the list contains the headers, so we separate it from the data
    cabeçalhos = lista[0].split(',')
    dados = [linha.split(',') for linha in lista[1:]]
    
    # Create a DataFrame with the data and specified headers
    dataframe = pd.DataFrame(dados, columns=cabeçalhos)
    
    # Save the DataFrame as a CSV file
    dataframe.to_csv(nome_arquivo_csv, index=False)  # index=False to not include the index in the CSV file

# Analyze important features in relation to metric loss related to the feature
def calcula_padroes(clf, kfold, tipo=md.TESTE):
    """
    Analyze important features in relation to metric loss.

    Args:
        clf (str): Classifier name.
        kfold (str): K-fold cross-validation identifier.
        tipo (str, optional): Type of data (e.g., 'Test'). Defaults to md.TESTE.

    Returns:
        None
    """
    caminho, pref, md_name = io.criarCaminho(clf, kfold)

    X_train, y_train, X_test, y_test = io.loadDatasets_ite01(pref, caminho)
    model, mi = md.loadModel(caminho, pref, md_name)
    words1 = pd.read_csv(f'{caminho}{p.path_if}{pref}{md_name}{io.arquivo_er}0.csv')['padrao'].to_list()
    words0 = pd.read_csv(f'{caminho}{p.path_if}{pref}{md_name}{io.arquivo_er}1.csv')['padrao'].to_list()

    X_test = pp.preProcessaTexto03(X_test)
    maior_tamanho = max(len(words0), len(words1))
    cabecalho = ['tp,pre,cl,i.acc,f.acc,d.acc,i.pre,f.pre,d.pre,i.f1,f.f1,d.f1,i.esp,f.esp,d.esp,i.tn,f.tn,d.tn,i.fn,f.fn,d.fn,i.sen,f.sen,d.sen,i.tp,f.tp,d.tp,i.fp,f.fp,d.fp,p.idx,p.qtd,a.idx,a.qtd,cat,cond,padrao']
    
    ces0, ces1 = ([] for _ in range(2))

    for i in tqdm(range(maior_tamanho)):
        if i < len(words0):
            saida0 = words0[i]
            ces0.extend(printCorrelacao01(saida0, 0, pref, tipo, mi, model, X_test, y_test))
        
        if i < len(words1):
            saida1 = words1[i]
            ces1.extend(printCorrelacao01(saida1, 1, pref, tipo, mi, model, X_test, y_test))
    
    cabecalho.extend(ces0)
    cabecalho.extend(ces1)
    #dataframe[nova_ordem] If you want to change the order of the columns

    caminho_if = f'{caminho}{p.path_if}'
    io.criarPasta(caminho_if)
    nome_arquivo = f'{caminho_if}{pref}{md_name}{io.arquivo_ce}.csv'
    lista_para_csv(cabecalho, nome_arquivo)

### MERGE FILES FROM EACH KFOLD
# Summarize Spurious Correlations
def sumarizar_ce(caminho, pref, md_name):
    """
    Summarize spurious correlations.

    Args:
        caminho (str): Base path to the files.
        pref (str): Prefix for the file names.
        md_name (str): Name of the model.

    Returns:
        None
    """
    def find_cl_value(palavra):
        # Find the 'cl' value for the first occurrence of 'palavra' in the dataframe true_df
        cl_value = cond_true_df[cond_true_df['padrao'] == palavra]['cl'].iloc[0]
        return cl_value
    
    nome_arquivo = f'{caminho}{p.path_if}{pref}{md_name}{io.arquivo_ce}'
    nome_arquivo_final = f'{caminho}{p.path_if}{pref}{md_name}{io.arquivo_ce_resumido}'
    # Reading the CSV file
    df = pd.read_csv(f'{nome_arquivo}.csv')

    # Step 2: Split the dataframe into two based on the 'cond' column value and create CSV files
    cond_true_df = df[df['cond'] == True]
    cond_false_df = df[df['cond'] == False]

    # Perform the correct aggregation
    aggregated_df_only_palavras = df.groupby('padrao', as_index=False).agg(
        cl=('cl', 'first'),
        padrao=('padrao', 'first'),
        cat=('cat', 'first'),
        pIdx=('p.idx', 'sum'),
        pQtd=('p.qtd', 'sum'),
        aIdx=('a.idx', 'sum'),
        aQtd=('a.qtd', 'sum'),
    )
    
    # First, get a dataframe with 'palavras', 'iceTotal', and 'qtdTotal' ordered by 'qtdTotal'
    final_df_ordered = aggregated_df_only_palavras

    # Apply the function to create the 'cl' column in the final_df_ordered dataframe
    final_df_ordered['cl'] = final_df_ordered['padrao'].apply(find_cl_value)
    #final_df_ordered['iceTotal'] = round(final_df_ordered['iceTotal'], 3)

    final_df_ordered.to_csv(f'{caminho}{p.path_if}{pref}{md_name}Temp00.csv')

    nova_ordem = ['cl', 'padrao', 'cat', 'pIdx', 'pQtd', 'aIdx', 'aQtd']
    final_df_ordered = final_df_ordered[nova_ordem]

    final_df_0 = final_df_ordered[final_df_ordered['cl'] == 0]
    final_df_1 = final_df_ordered[final_df_ordered['cl'] == 1]

    final_df_0.to_csv(f'{nome_arquivo_final}0.csv', index=False)
    final_df_1.to_csv(f'{nome_arquivo_final}1.csv', index=False)

# Summarize Feature Importance
def sumarizar_if(caminho, pref, md_name):
    """
    Summarize feature importance.

    Args:
        caminho (str): Base path to the files.
        pref (str): Prefix for the file names.
        md_name (str): Name of the model.

    Returns:
        None
    """
    f_global = fi.lerIf(caminho, pref, md_name, 'Lime')
    
    f_global[0] = dict(sorted(f_global[0].items(), key=itemgetter(1), reverse=True))
    f_global[1] = dict(sorted(f_global[1].items(), key=itemgetter(1), reverse=True))
    f_global[0], f_global[1] = ife.filter_common_keys_by_max_value(f_global[0], f_global[1])

    io.gravarJson(f'{caminho}{p.path_if}{pref}{md_name}{io.arquivo_if_resumido}0.json', f_global[0])
    io.gravarJson(f'{caminho}{p.path_if}{pref}{md_name}{io.arquivo_if_resumido}1.json', f_global[1])

# Summarize Feature Errors
def sumarizar_ef(caminho, pref, md_name):
    """
    Summarize feature errors.

    Args:
        caminho (str): Base path to the files.
        pref (str): Prefix for the file names.
        md_name (str): Name of the model.

    Returns:
        None
    """
    def sumarize_er(caminho, pref, md_name, classe):
        """
        Summarize errors for a specific class.

        Args:
            caminho (str): Base path to the files.
            pref (str): Prefix for the file names.
            md_name (str): Name of the model.
            classe (int): Class label (0 or 1).

        Returns:
            None
        """
        # Load data from the CSV file
        df = pd.read_csv(f'{caminho}{p.path_if}{pref}{md_name}{io.arquivo_er}{classe}.csv')

        # Group data by the first column (assuming it is an identification or category column)
        # and sum the columns 'freqTreinoMesmaClasse' and 'freqErroTesteMesmaClasse'
        df_agrupado = df.groupby('padrao', as_index=False).agg({
            'freqTreinoMesmaClasse': 'sum',
            'freqTreinoOutraClasse': 'sum',
            'freqErroTesteMesmaClasse': 'sum',
            'freqErroTesteOutraClasse': 'sum'
        })

        # Add a new column 'div_frequenciasOM' which is the ratio of 'freqTreinoOutraClasse' to 'freqTreinoMesmaClasse'
        df_agrupado['div_frequenciasOM'] = round(df_agrupado['freqTreinoOutraClasse'] / df_agrupado['freqTreinoMesmaClasse'], 3)
        # Add a new column 'div_freqErroOM' which is the ratio of 'freqErroTesteOutraClasse' to 'freqErroTesteMesmaClasse'
        df_agrupado['div_freqErroOM'] = round(df_agrupado['freqErroTesteOutraClasse'] / df_agrupado['freqErroTesteMesmaClasse'], 3)

        # Filter the dataframe to include only rows where 'freqErroTesteMesmaClasse' is greater than or equal to 3
        df_filtrado = df_agrupado[df_agrupado['freqErroTesteMesmaClasse'] >= 3]

        # Save the result to another CSV file
        caminho_arquivo_saida = f'{caminho}{p.path_if}{pref}{md_name}{io.arquivo_er_resumido}{classe}.csv'
        df_filtrado.to_csv(caminho_arquivo_saida, index=False)

    sumarize_er(caminho, pref, md_name, 0)
    sumarize_er(caminho, pref, md_name, 1)

# Merge Data from Different K-Folds
def juntarDados(clf, classe=None, tipo=None):
    """
    Merge data from different K-Folds.

    Args:
        clf (str): Classifier name.
        classe (int, optional): Class label (0 or 1). Defaults to None.
        tipo (str, optional): Type of data ('if' or 'ef'). Defaults to None.

    Returns:
        None
    """
    dados = []
    for kfold in tqdm(range(1, 6)):
        caminho, pref, md_name = io.criarCaminho(clf, kfold)
        if classe is None:
            # Treatment for CE
            arquivos = [f'{caminho}{p.path_if}{pref}{md_name}{io.arquivo_ce}.csv']
            tipo_arquivo = 'csv'
        else:
            # Treatment for IF and EF
            if tipo == 'if':
                arquivos = [f'{caminho}{p.path_if}{pref}{md_name}{io.arquivo_if}{classe}.json']
                tipo_arquivo = 'json'
            else:  # tipo == 'ef'
                arquivos = [f'{caminho}{p.path_if}{pref}{md_name}{io.arquivo_er}{classe}.csv']
                tipo_arquivo = 'csv'
        
        for arquivo in arquivos:
            if os.path.exists(arquivo):
                if tipo_arquivo == 'csv':
                    df = pd.read_csv(arquivo)
                    dados.append(df)
                elif tipo_arquivo == 'json':
                    with open(arquivo, 'r') as f:
                        lista = json.load(f)
                        dados.extend(lista)
    
    caminho_validacao, pref_validacao, md_name = io.criarCaminho(clf, 6)
    io.criarPasta(caminho_validacao) 
    io.criarPasta(f'{caminho_validacao}{p.path_if}')
    
    if classe is None:
        arquivo_final = f'{caminho_validacao}{p.path_if}{pref_validacao}{md_name}{io.arquivo_ce}.csv'
        dados_concatenados = pd.concat(dados)
        dados_concatenados = dados_concatenados.sort_values(by='padrao', ascending=True)
        dados_concatenados_false = dados_concatenados[dados_concatenados['cond'] == False]
        dados_concatenados = dados_concatenados[dados_concatenados['cond'] == True]

        dados_concatenados.to_csv(arquivo_final, index=False)

        position = len(arquivo_final) - 4
        arquivo_final_false = arquivo_final[:position] + '_false' + arquivo_final[position:]
        arquivo_final_mp = arquivo_final[:position] + '_mp' + arquivo_final[position:]
        dados_concatenados_false.to_csv(arquivo_final_false, index=False)
    else:
        if tipo == 'if':
            arquivo_final = f'{caminho_validacao}{p.path_if}{pref_validacao}{md_name}{io.arquivo_if}{classe}.json'
            io.gravarJson(arquivo_final, dados)
        else:  # tipo == 'ef'
            arquivo_final = f'{caminho_validacao}{p.path_if}{pref_validacao}{md_name}{io.arquivo_er}{classe}.csv'
            dados_concatenados = pd.concat(dados)
            dados_concatenados.to_csv(arquivo_final, index=False)

# Move auxiliary files for calculations
def mover_arquivos(caminho, pref, md_name):
    """
    Move auxiliary files for calculations to a subfolder.

    Args:
        caminho (str): Base path to the files.
        pref (str): Prefix for the file names.
        md_name (str): Name of the model.

    Returns:
        None
    """
    # Path of the folder where the files are located
    caminho_pasta = f'{caminho}{p.path_if}'
    
    # Path of the subfolder where the files should be moved
    caminho_subpasta = os.path.join(caminho_pasta, 'todos')
    pa = f'{pref}{md_name}'
    # List of file names that should not be moved
    arquivos_excluir = [f'{pa}{io.arquivo_ce}.csv', f'{pa}{io.arquivo_if}0.json', f'{pa}{io.arquivo_if}1.json',
                        f'{pa}{io.arquivo_er}0.csv', f'{pa}{io.arquivo_er}1.csv']
    
    # Create the subfolder if it does not exist
    if not os.path.exists(caminho_subpasta):
        os.makedirs(caminho_subpasta)
    
    # Iterate through all files in the folder
    for arquivo in os.listdir(caminho_pasta):
        # Check if the current file is not in the exclusion list and is a file (not a folder)
        if arquivo in arquivos_excluir and os.path.isfile(os.path.join(caminho_pasta, arquivo)):
            # Full path of the current file
            caminho_completo_arquivo = os.path.join(caminho_pasta, arquivo)
            
            # Full path of the destination file in the subfolder
            destino_arquivo = os.path.join(caminho_subpasta, arquivo)
            
            # Move the file to the subfolder
            shutil.move(caminho_completo_arquivo, destino_arquivo)
