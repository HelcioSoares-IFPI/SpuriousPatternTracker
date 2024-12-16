import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import os
import warnings
import utils.path as p
from tqdm.auto import tqdm

# Disable parallelism warnings for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

def preparar_dataset(encodings, labels):
    """
    Prepare a PyTorch dataset with encodings and labels.

    Parameters:
    - encodings: Encoded input data.
    - labels: Corresponding labels for the data.

    Returns:
    - Dataset: A PyTorch Dataset object.
    """
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)
    
    return Dataset(encodings, labels)

def retirar_palavras_das_sentencas(word, sentence_list):
    """
    Remove specific words from a list of sentences.

    Parameters:
    - word: Word or phrase to remove.
    - sentence_list: List of sentences to process.

    Returns:
    - result: List of sentences with the specified word removed.
    """
    set_a = set(word.split())
    result = []
    for sentence in sentence_list:
        list_b = sentence.split()
        if set_a.issubset(set(list_b)):
            result_list = [w for w in list_b if w not in set_a]
            result.append(' '.join(result_list))
        else:
            result.append(sentence)
    return result

def encode_data(tokenizer, df):
    """
    Encode text data using a tokenizer.

    Parameters:
    - tokenizer: Pre-trained tokenizer.
    - df: DataFrame containing the text data.

    Returns:
    - Encoded text data.
    """
    return tokenizer(df['texto'].tolist(), padding=True, truncation=True, return_tensors="pt", max_length=512)

def perturbar_avaliar_treino(word, original_texts, test_data, tokenizer, model, diretorioLogs, diretorioOutputs):
    """
    Perturb the training text and evaluate the model.

    Parameters:
    - word: Word or phrase to perturb.
    - original_texts: Original text data.
    - test_data: Test dataset.
    - tokenizer: Pre-trained tokenizer.
    - model: Pre-trained model for evaluation.
    - diretorioLogs: Directory for log files.
    - diretorioOutputs: Directory for output files.

    Returns:
    - perturbed_results: Evaluation results of the perturbed data.
    """
    perturbed_texts = retirar_palavras_das_sentencas(word, original_texts)
    perturbed_data = pd.DataFrame({'texto': perturbed_texts, 'classe': test_data['classe']})
    perturbed_encodings = encode_data(tokenizer, perturbed_data)
    perturbed_dataset = preparar_dataset(perturbed_encodings, perturbed_data['classe'].tolist())
    perturbed_results = avaliar(perturbed_dataset, model, diretorioLogs, diretorioOutputs)
    return perturbed_results

def compute_metrics(pred):
    """
    Compute metrics for model evaluation.

    Parameters:
    - pred: Predictions from the model.

    Returns:
    - metrics: Dictionary with accuracy, precision, recall, F1, sensitivity, specificity, and confusion matrix components.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    pre, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sen = tp / (tp + fn)
    esp = tn / (tn + fp)
    return {'acc': acc, 'pre': pre, 'f1': f1, 'esp': esp, 'tn': tn, 'fn': fn, 'sen': sen, 'tp': tp, 'fp': fp}

def avaliar(dataset: torch.utils.data.Dataset, modelo: AutoModelForSequenceClassification, diretorioLogs: str, diretorioOutputs: str):
    """
    Evaluate the model using the given dataset.

    Parameters:
    - dataset: Dataset for evaluation.
    - modelo: Pre-trained model for evaluation.
    - diretorioLogs: Directory for log files.
    - diretorioOutputs: Directory for output files.

    Returns:
    - resultados: Evaluation results.
    """
    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        logging_dir=diretorioLogs,
        logging_steps=10,
        save_total_limit=1,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        output_dir=diretorioOutputs,
    )

    trainer = Trainer(
        model=modelo,
        args=training_args,
        compute_metrics=compute_metrics,
    )

    resultados = trainer.evaluate(dataset)
    return resultados

def avaliar_perturbacao(fold: int, diretorioFold: str, device: torch.device, diretorioLogs: str, diretorioOutputs: str):
    """
    Evaluate the model with perturbed data.

    Parameters:
    - fold: Fold identifier for cross-validation.
    - diretorioFold: Directory containing fold-specific data.
    - device: Torch device (CPU or GPU).
    - diretorioLogs: Directory for log files.
    - diretorioOutputs: Directory for output files.

    Saves results of the perturbed data evaluation.
    """
    model_path = os.path.join(diretorioFold, p.p_mod)

    eval_ = p.lerJson(os.path.join(diretorioFold, p.p_res, f'{fold}BertMetricas.json'))
    metricasIniciais = eval_['Teste']

    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    test_data = pd.read_csv(os.path.join(diretorioFold, f'{fold}Teste_.csv'))
    original_texts = test_data['texto'].tolist()

    resultados_perturbados0 = {}
    resultados_perturbados1 = {}

    words1 = pd.read_csv(os.path.join(diretorioFold, p.p_inf, f'{fold}bertErrorFeatures1.csv'))['padrao'].to_list()
    words0 = pd.read_csv(os.path.join(diretorioFold, p.p_inf, f'{fold}bertErrorFeatures0.csv'))['padrao'].to_list()

    maior_tamanho = max(len(words0), len(words1))
    for i in tqdm(range(maior_tamanho), desc=f'Perturbation for fold {fold}', total=maior_tamanho, colour='blue'):
        if i < len(words0):
            word0 = words0[i]
            perturbed_results0 = perturbar_avaliar_treino(word0, original_texts, test_data, tokenizer, model, diretorioLogs, diretorioOutputs)
            resultados_perturbados0[word0] = perturbed_results0

        if i < len(words1):
            word1 = words1[i]
            perturbed_results1 = perturbar_avaliar_treino(word1, original_texts, test_data, tokenizer, model, diretorioLogs, diretorioOutputs)
            resultados_perturbados1[word1] = perturbed_results1

        if (i + 1) % 10 == 0:
            p.gravarJson(os.path.join(diretorioFold, p.p_inf, f'{fold}bertPerturbacao.json'), [resultados_perturbados0, resultados_perturbados1, metricasIniciais])

def pertubarTeste(diretorio: str, fold: int) -> None:
    """
    Perturb the test dataset and evaluate the model for a specific fold.

    Parameters:
    - diretorio: Directory containing data.
    - fold: Fold identifier for cross-validation.

    This function manages directories and calls `avaliar_perturbacao` for evaluation.
    """
    print(f"\nPerturbing test data for fold: {fold}\n")
    device = torch.device("cuda")
    diretorioFold = os.path.join(diretorio, fold)
    diretorioLogs = os.path.join(diretorioFold, p.p_log)
    p.verificarCaminho(diretorioLogs)
    diretorioOutputs = os.path.join(diretorioFold, p.p_out)
    p.verificarCaminho(diretorioOutputs)
    avaliar_perturbacao(fold, diretorioFold, device, diretorioLogs, diretorioOutputs)
