from nltk.probability import FreqDist
from enum import Enum

def freqDist(corpus):
    """
    Calculate the frequency distribution of words in the corpus.
    """
    textList = ' '.join(corpus).split(' ')
    return FreqDist(textList)

def showDict(label, textos, qtd_tokens):
    """
    Display the frequency distribution of words in the texts.
    """
    fdist = freqDist(textos)
    print(label, len(fdist))
    fdist.plot(qtd_tokens)

class Ce(Enum):
    """
    Enumeration for different CE types.
    """
    Cn  = 'Cn'
    Cnn = 'Cnn'
    Cnm = 'Cnm'

def retirar_CE(X_, ces):
    """
    Remove specified CE tokens from the list of sentences.
    """
    for i in range(len(X_)):
        for token in ces:
            X_[i] = X_[i].replace(token, ' ').lstrip()
    return X_

def ctp(tempo_inicial, tempo_final):
    """
    Calculate the time difference and return it in a human-readable format.
    """
    # Calculate the time difference
    diferenca = tempo_final - tempo_inicial
    
    # Convert the difference to seconds, minutes, and hours
    segundos = diferenca
    minutos = segundos / 60
    horas = minutos / 60
    
    # Decide the return format based on the duration
    if segundos < 60:
        return f"{segundos:.2f} seconds"
    elif minutos < 60:
        return f"{minutos:.2f} minutes"
    else:
        return f"{horas:.2f} hours"
