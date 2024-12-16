"""
This script trains a BERT model for binary classification and evaluates its performance.
It includes functionality for:
- Data encoding using a tokenizer.
- Training and evaluation of a BERT model.
- Calculation of evaluation metrics such as accuracy, precision, recall, and F1-score.
- Identification and storage of misclassified examples from the training and testing datasets.
- Saving the trained model and tokenizer for future use.

The model is trained and evaluated using the Hugging Face `transformers` library.
"""

import torch
import os
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import utils.path as p
import warnings
import json
from utils.path import verificarCaminho

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Dataset(torch.utils.data.Dataset):
    """
    A custom Dataset class to handle tokenized data and labels for PyTorch.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Retrieves a single item from the dataset at the specified index.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        Returns the length of the dataset (number of samples).
        """
        return len(self.labels)


def compute_metrics(pred):
    """
    Computes evaluation metrics for the model, including:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - Sensitivity and specificity (calculated from the confusion matrix)

    Parameters:
    - pred: Prediction object from the Hugging Face Trainer.

    Returns:
    - A dictionary containing the evaluation metrics.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    pre, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    # Calculating the confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    # Calculating sensitivity and specificity
    sen = tp / (tp + fn)
    esp = tn / (tn + fp)
    return {'acc': acc, 'pre': pre, 'f1': f1, 'esp': esp, 'tn': tn, 'fn': fn, 'sen': sen, 'tp': tp, 'fp': fp}


def encode_data(df: pd.DataFrame, tokenizer: AutoTokenizer):
    """
    Encodes the dataset using the tokenizer.

    Parameters:
    - df: A pandas DataFrame containing a column named 'texto'.
    - tokenizer: The tokenizer object from the Hugging Face library.

    Returns:
    - Encoded data ready for model input.
    """
    return tokenizer(df['texto'].tolist(), padding=True, truncation=True, return_tensors="pt", max_length=512)


def get_misclassified_data(dataset: Dataset, trainer: Trainer, tokenizer: AutoTokenizer):
    """
    Retrieves misclassified examples from the dataset.

    Parameters:
    - dataset: The PyTorch Dataset object.
    - trainer: The Hugging Face Trainer object.
    - tokenizer: The tokenizer object used to decode the input IDs.

    Returns:
    - misclassified_texts: List of misclassified sentences.
    - true_misclassified_labels: True labels of the misclassified sentences.
    - pred_misclassified_labels: Predicted labels of the misclassified sentences.
    """
    predictions = trainer.predict(dataset)
    preds = predictions.predictions.argmax(-1)
    true_labels = predictions.label_ids

    misclassified_indices = [i for i, (true, pred) in enumerate(zip(true_labels, preds)) if true != pred]
    misclassified_texts = [dataset[i]['input_ids'] for i in misclassified_indices]
    misclassified_texts = [tokenizer.decode(text, skip_special_tokens=True) for text in misclassified_texts]

    true_misclassified_labels = [true_labels[i] for i in misclassified_indices]
    pred_misclassified_labels = [preds[i] for i in misclassified_indices]

    return misclassified_texts, true_misclassified_labels, pred_misclassified_labels


def trainModel(fold: int, diretorio: str, tokenizer: AutoTokenizer, device) -> None:
    """
    Trains a BERT model for a specific fold and evaluates its performance.

    Parameters:
    - fold: The current fold number.
    - diretorio: Directory containing the training and testing datasets.
    - tokenizer: The tokenizer object for encoding text.
    - device: The device (CPU/GPU) for training.

    Returns:
    - None.
    """
    diretorioFold = os.path.join(diretorio, str(fold))

    train_data = pd.read_csv(os.path.join(diretorioFold, f'{fold}Treino_.csv'))
    test_data = pd.read_csv(os.path.join(diretorioFold, f'{fold}Teste_.csv'))

    train_encodings = encode_data(train_data, tokenizer)
    test_encodings = encode_data(test_data, tokenizer)

    train_dataset = Dataset(train_encodings, train_data['classe'].tolist())
    test_dataset = Dataset(test_encodings, test_data['classe'].tolist())

    model = AutoModelForSequenceClassification.from_pretrained(
        'google-bert/bert-base-uncased', 
        num_labels=len(train_data['classe'].unique())
    ).to(device)
    model.resize_token_embeddings(len(tokenizer))

    diretorioLogs = os.path.join(diretorioFold, p.p_log)
    verificarCaminho(diretorioLogs)
    
    diretorioOutputs = os.path.join(diretorioFold, p.p_out)
    verificarCaminho(diretorioOutputs)

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
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate on training and testing sets
    train_results = trainer.evaluate(train_dataset)
    test_results = trainer.evaluate(test_dataset)

    print("Training Results:", train_results)
    print("Testing Results:", test_results)

    results_dict = {'Training': train_results, 'Testing': test_results}

    # Save results as JSON
    diretorioResults = os.path.join(diretorioFold, p.p_res)
    verificarCaminho(diretorioResults)
    json.dump(
        results_dict, 
        open(os.path.join(diretorioResults, f'{fold}BertMetrics.json'), 'w'), 
        indent=4, ensure_ascii=False
    )

    # Save the model and tokenizer
    diretorioModelo = os.path.join(diretorioFold, p.p_mod)
    verificarCaminho(diretorioModelo)

    model.save_pretrained(diretorioModelo)
    tokenizer.save_pretrained(diretorioModelo)

    # Save misclassified examples for training
    train_texts, train_true_labels, train_pred_labels = get_misclassified_data(train_dataset, trainer, tokenizer)
    pd.DataFrame({
        'texto': train_texts,
        'classe_real': train_true_labels,
        'classe_predita': train_pred_labels
    }).to_csv(os.path.join(diretorioFold, 'misclassified_train.csv'), index=False)

    # Save misclassified examples for testing
    test_texts, test_true_labels, test_pred_labels = get_misclassified_data(test_dataset, trainer, tokenizer)
    pd.DataFrame({
        'texto': test_texts,
        'classe_real': test_true_labels,
        'classe_predita': test_pred_labels
    }).to_csv(os.path.join(diretorioFold, 'misclassified_test.csv'), index=False)


def treinarBert(diretorio: str, fold: str) -> None:
    """
    Trains the BERT model for a specific fold and saves all relevant outputs.

    Parameters:
    - diretorio: Directory containing the datasets.
    - fold: Current fold number.

    Returns:
    - None.
    """
    print(f'\nTraining BERT model for fold: {fold}\n')
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
    trainModel(fold, diretorio, tokenizer, device)
