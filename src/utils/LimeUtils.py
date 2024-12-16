# Importing packages
from utils import io as io
from nltk.stem.porter import *
from lime.lime_text import LimeTextExplainer

# Variables for the task
class_names = ['Não aquisição', 'Aquisição']
explainer = LimeTextExplainer(class_names=class_names, verbose=False)

# Explain locally and return the weights of each word
def localExplain_v(explainer, model, text_to_explain, print_):
    """
    Explain a text instance locally and return the weights of each word.

    Args:
        explainer (LimeTextExplainer): The LIME text explainer.
        model (object): The model to explain.
        text_to_explain (str): The text to explain.
        print_ (bool): Whether to print the explanation.

    Returns:
        list: Two dictionaries containing the weights of words for each class.
    """
    exp = explainer.explain_instance(text_to_explain, model.predict_proba, labels=[0, 1], num_features=len(text_to_explain))
    if print_:
        exp.show_in_notebook(text=True, labels=[1])
        
    lime_vals = exp.as_list(label=1)
    f0_local = dict()
    f1_local = dict()
    
    for tupla in lime_vals:
        if tupla[1] < 0:
            f0_local[tupla[0]] = abs(tupla[1])
        else:
            f1_local[tupla[0]] = tupla[1]
            
    return [f0_local, f1_local]
 
    