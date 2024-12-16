"""
This script executes the complete pipeline for detecting spurious patterns using Explainable Artificial Intelligence (XAI) techniques and unsupervised learning.

The pipeline is applied for each fold of the training and testing datasets, leveraging the following modules:
- `createTrainTest`: Creates training and testing datasets.
- `trainBERT`: Trains the BERT model.
- `explainLimeGlobal`: Generates global explanations for model predictions using LIME.
- `generateIF`: Generates Feature Importances (IF).
- `disturbTest`: Perturbs the testing dataset and evaluates the model's response.
- `calculateConf`: Identifies confounders, which are potential spurious patterns affecting the model's decisions.
- `calculatePatterns`: Extracts patterns identified as significant by the models.
- `gerarResultados`: Generates the final results after processing the pipeline.

The script iterates over 5 folds of the data split, using the `tqdm` library to display progress.
"""

from .utils.BERTcreateTrainTest     import createDatasets
from .utils.BERTtrain               import treinarBert
from .utils.BERTexplainLimeGlobal   import explicarLimeGlobal
from .utils.BERTgenerateIF          import gerarIF
from .utils.BERTdisturbTest         import pertubarTeste
from .utils.BERTgerarResultados     import gerar_resultados
from .utils.BERTcalculatePatterns   import calculaPadroes
from .utils.BERTcalculateConf       import calcularConf
from  tqdm.auto                      import tqdm

def main():
    """
    Main function to execute the complete pipeline for detecting spurious patterns in binary classification datasets.

    The function iterates through 5 folds, performing model training, explainability analysis (LIME),
    feature importance generation, error analysis, testing dataset perturbation, pattern extraction, 
    and confounder detection. Finally, it generates the overall results.

    Parameters:
    - None.

    Returns:
    - None.
    """
    
    DIRETORIO = ''  # Directory where the training and testing datasets are stored, separated by fold.
    print('\nRunning the complete pipeline...\n')

    # Iterates through the 5 folds of training and testing
    for fold in tqdm(range(1, 6), desc='Running the complete pipeline...', colour='red'):
        fold = str(fold)

        # Train the BERT model
        treinarBert(DIRETORIO, fold)

        # Generate global explanations for predictions using LIME
        explicarLimeGlobal(DIRETORIO, fold)

        # Generate Feature Importances (IF)
        gerarIF(DIRETORIO, fold)

        # Identify confounders (potential spurious patterns)
        calcularConf(DIRETORIO, fold)

        # Perturb the testing dataset for impact analysis
        pertubarTeste(DIRETORIO, fold)

        # Extract patterns identified as significant by the models
        calculaPadroes(DIRETORIO, fold)

    # Generate the final results of the experiment
    gerar_resultados(DIRETORIO)


if __name__ == '__main__':
    main()
