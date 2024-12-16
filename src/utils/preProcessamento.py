import re
import nltk.corpus
import pandas as pd
from nltk.tokenize import word_tokenize
import unidecode 
from utils import io       as io
from utils import path     as p
import json

# Load Portuguese stopwords from NLTK
stopwords = nltk.corpus.stopwords.words('portuguese')

# Load city-specific stopwords from a JSON file
stop_words_cidade = json.load(open(p.path_list+'cidades.json', encoding='UTF-8'))

# Load context-specific stopwords from a CSV file
stop_word_contexto = pd.read_csv(p.path_list + 'stop_words_contexto.csv')['stop_word'].to_list()

# List of special characters to be removed
specialChars = ['nº','nºs','n°','n.º','n.',':','–','n °',',','.','“','”',';','º','°','%','’','‘','''''',#
                  ']','[','ª','"','(',')','\n','§','e/ou','<', '>']

# Initialize the Portuguese stemmer
stemmer = nltk.stem.RSLPStemmer()

def removeCaracteresEspeciais(linha):
    """
    Remove special characters from the line.
    """
    for i in range(len(specialChars)):
        linha = linha.replace(specialChars[i],'')
        
    linha = linha.replace('/', ' ')
    linha = linha.replace('-', ' ')
    return linha

def removePalavrasMenorQue3(texto):
    """
    Remove words with less than 3 characters from the text.
    """
    # Split the text into words
    palavras = texto.split()
    
    # Filter words with more than 3 characters
    palavras_filtradas = [palavra for palavra in palavras if len(palavra) > 2]
    
    # Join the filtered words into a new string
    novo_texto = ' '.join(palavras_filtradas)
    return novo_texto

def removeTokens(listaTokens, listaTokensRemover):
    """
    Remove specified tokens from the token list.
    """
    return [token for token in listaTokens if token not in listaTokensRemover]

def trocarPorSinonimo(texto, lista_expressoes, lista_sinonimos):
    """
    Replace expressions in the text with their synonyms.
    """
    for i in range(len(lista_sinonimos)):
        texto = texto.replace(lista_expressoes[i], lista_sinonimos[i])
    return texto

def trocarPadraoNo(texto):
    """
    Replace patterns like '1o', '2o' with '1', '2' in the text.
    """
    padroes = re.findall("\d+o", texto)
    if len(padroes) != 0:
        for padrao in padroes:
            texto = texto.replace(padrao, padrao[0:len(padrao)-1])
    return texto

def trocarPadraoRepeticao(pattern, texto):
    """
    Replace repeated patterns in the text with a space.
    """
    padroes = re.findall(pattern, texto)
    if len(padroes) != 0:
        for padrao in padroes:
            texto = texto.replace(padrao, ' ')
    return texto

def trocarPadraoValor(texto):
    """
    Replace monetary values in the text with the symbol 'Σ'.
    """
    padroes = re.finditer("(r\$( )*)?(\d{1,3}(\.\d{3})*|\d+)(\,\d{2})+", texto, flags=re.IGNORECASE)
    for padrao in padroes:
        texto = texto.replace(padrao.group(), 'Σ')
    return texto

def preProcessaTexto01(texto):
    """
    Preprocess the text by applying various transformations.
    """
    texto = texto.replace('\n', ' ')
    texto = texto.lower()

    # Remove patterns
    texto = trocarPadraoNo(texto)  
    
    # Remove numbers
    texto = trocarPadraoRepeticao(r'\d+', texto)    
    
    texto = trocarPadraoValor(texto)
    
    texto = trocarPadraoRepeticao(r'(\.)+', texto)
    texto = trocarPadraoRepeticao(r'(\_)+', texto)

    texto = removePalavrasMenorQue3(texto)

    # Preprocessing at the level of expressions or characters
    texto = removeCaracteresEspeciais(texto)

    # Remove accents from words
    texto = unidecode.unidecode(texto) 

    # Preprocessing at the token level
    tokens = word_tokenize(texto)
    tokens = removeTokens(tokens, stopwords)

    retorno = ' '.join(tokens)
    return retorno

def getStopWordContexto():
    """
    Retrieve context-specific stopwords from a CSV file.
    """
    df = pd.read_csv(p.path_list + 'stop_words_contexto.csv')
    return df['stop_word'].tolist()

def preProcessaTexto03(textos):
    """
    Preprocess a list of texts by applying various transformations.
    """
    for i in range(len(textos)):
        textos[i] = textos[i].replace('agua branca', '')
        texto = preProcessaTexto01(textos[i])    
        tokens = word_tokenize(texto)   
        tokens = removeTokens(tokens, stop_words_cidade)
        tokens = removeTokens(tokens, stop_word_contexto)
        textos[i] = ' '.join(tokens)
    return textos

def retirar_mais_frequentes(X_, l_mais_freq):
    """
    Remove the most frequent words from the list of sentences.
    """
    for i in range(len(X_)):
        # Split the sentence into words
        palavras = X_[i].split()

        # Replace only the words that are equal to the most frequent words
        palavras = [palavra for palavra in palavras if palavra not in l_mais_freq]

        # Join the words back into a sentence without extra spaces
        X_[i] = ' '.join(palavras)
    return X_

def retirar_mais_frequentes__(X_, l_mais_freq):
    """
    Remove the most frequent words from the list of sentences using a set for quick lookup.
    """
    l_mais_freq_set = set(l_mais_freq)
    X_ = [' '.join(palavra for palavra in x.split() if palavra not in l_mais_freq_set) for x in X_]
    return X_

def retirar_mais_frequentes_(X_, l_mais_freq):
    """
    Remove the most frequent words from the list of sentences using a set for quick lookup.
    """
    set_mais_freq = set(l_mais_freq)
    X_ = [' '.join(palavra for palavra in x.split() if palavra not in set_mais_freq) for x in X_]
    return X_


def remover_expressoes(sentenca, lista_expressoes):
    """
    Remove specified expressions from the sentence.
    """
    # Split the sentence into words
    palavras_sentenca = sentenca.split()

    # Lists to store removed expressions and remaining words
    expressoes_encontradas = []
    palavras_restantes = palavras_sentenca.copy()

    # Check each expression in the list
    for expressao in lista_expressoes:
        palavras_expressao = expressao.split()
        # Check if all words of the expression are in the sentence
        if all(palavra in palavras_sentenca for palavra in palavras_expressao):
            expressoes_encontradas.append(expressao)
            # Remove the words of the found expression from the list of remaining words
            for palavra in palavras_expressao:
                if palavra in palavras_restantes:
                    palavras_restantes.remove(palavra)

    # Reconstruct the sentence without the words of the removed expressions
    sentenca_modificada = ' '.join(palavras_restantes)

    return sentenca_modificada, expressoes_encontradas

def remover_palavras(sentenca, lista_palavras):
    """
    Remove specified words from the sentence.
    """
    # Split the sentence into words
    palavras_sentenca = sentenca.split()

    # Lists to store removed words and remaining words
    palavras_encontradas = []
    palavras_restantes = []

    # Check each word in the sentence
    for palavra in palavras_sentenca:
        if palavra in lista_palavras:
            palavras_encontradas.append(palavra)
        else:
            palavras_restantes.append(palavra)

    # Reconstruct the sentence without the removed words
    sentenca_modificada = ' '.join(palavras_restantes)

    return sentenca_modificada, palavras_encontradas

def remove_ce(sentenca, ce_bi, ce_uni):
    """
    Remove specified expressions and words from the sentence.
    """
    ce_sentences = []
    sentenca_modificada, expressoes_removidas = remover_expressoes(sentenca, ce_bi)
    sentenca_modificada, palavras_removidas = remover_palavras(sentenca_modificada, ce_uni)
    ce_sentences = expressoes_removidas + palavras_removidas
    return sentenca_modificada, ce_sentences

def remove_ce_list(X, ce_bi, ce_uni):
    """
    Remove specified expressions and words from a list of sentences.
    """
    ce = []
    for i in range(len(X)):
        X[i], ce = remove_ce(X[i], ce_bi, ce_uni)
    return X
