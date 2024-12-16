# SpuriousPatternTracker

The **SpuriousPatternTracker** implements the methodology presented in the article *"Detection of Spurious Correlations in Public Procurement Descriptions Using Explainable Artificial Intelligence and Unsupervised Learning"*. This project focuses on identifying spurious patterns in Natural Language Processing (NLP) tasks, particularly for binary classification.

Artificial Intelligence (AI) models, including Deep Learning, linear, probabilistic, and rule-based or optimization models, often operate as black boxes due to limited transparency and explainability, which generates uncertainty in decision-making processes. This study addresses spurious correlations, defined as associations between patterns and classes that do not reflect causal relationships, affecting AI models' reliability and applicability. In Natural Language Processing (NLP), these correlations lead to inaccurate predictions, bias, and challenges in model generalization. We propose a methodology that employs Explainable Artificial Intelligence (XAI) techniques to detect spurious patterns in textual datasets for binary classification tasks. The methodology uses the K-means algorithm to cluster patterns, interpreting them based on their distance to centroids under the hypothesis that more distant patterns have higher spuriousness. We applied this methodology to bidding and contract datasets from the Tribunal de Contas do Estado do Piauí (TCE-PI), based on Stochastic Gradient Descent, Logistic Regression, and BERT models. The results support the hypothesis that patterns farther from centroids exhibit higher spuriousness and demonstrate the clustering's consistency across models and datasets. The methodology operates independently of the techniques used in its stages, enabling automatic detection and quantification of spurious patterns without prior human intervention.

## Objectives
1. **Automated Detection**: Identify spurious patterns without prior human intervention.
2. **Spuriousness Quantification**: Measure the spuriousness of patterns using cluster distances.
3. **Interpretability**: Group patterns logically for better visualization and insights.
4. **Practical Validation**: Apply the methodology to public procurement datasets.

## Methodology
The methodology consists of a seven-stage pipeline:
1. **Preprocessing**: Clean text and represent it using TF-IDF (LRG/SGD) or embeddings (BERT).
2. **Model Training and Testing**: Train Logistic Regression (LRG), Stochastic Gradient Descent (SGD), and BERT models.
3. **Extract Important Words**: Identify relevant tokens using LIME, SHAP, and intrinsic explainers.
4. **Error Analysis**: Locate misclassified sentences and their similar counterparts in the training set.
5. **Pattern Extraction**: Identify word combinations contributing to classification errors.
6. **Spurious Pattern Identification**: Evaluate the impact of patterns on model performance.
7. **Clustering**: Group patterns using K-means and assess their spuriousness.

## Contributions
- **Spuriousness Metric**: A novel metric based on cluster distances.
- **Public Dataset Application**: Analysis on public contracts and procurement datasets.
- **Integrated Techniques**: Combines XAI and unsupervised learning for efficient spurious pattern detection.

## Features
- Text preprocessing tools for cleaning and representation.
- Model training and explanation support for global explainers.
- Clustering and visualization for spurious pattern identification.
- Evaluation of pattern impacts on specificity and sensitivity metrics.

---

## Project Structure

### Fold Generator Script

##### Main File
- **`CreateDataset.py`**: This script is part of the **SpuriousPatternTracker** pipeline. It preprocesses datasets, initializes error tracking files, and generates k-fold splits for robust cross-validation in spurious pattern detection tasks.

---

##### Features

1. **Dataset Preprocessing**:
   - Standardizes raw text data by applying preprocessing and saving class-specific datasets (`classe0.csv` and `classe1.csv`).

2. **Error Tracking**:
   - Creates empty CSV files for each model to track classification errors (`<ModelName>Erros.csv`).

3. **k-Fold Generation**:
   - Splits datasets into balanced training and testing sets for k-fold cross-validation.

---

#### Workflow Overview

1. **Preprocess Dataset**:
   - Standardizes the input text and class labels, creating preprocessed datasets.

2. **Create Error Files**:
   - Initializes empty files to store model error information.

3. **Generate k-Folds**:
   - Creates `k` balanced folds, ensuring equal representation of class labels in training and testing sets.

---

#### Usage Instructions

1. **Prepare Input**:
   - Ensure the input file contains `text` (raw text) and `class` (labels: 0 or 1) columns.

2. **Configure Utilities**:
   - Verify paths and utilities in the `utils` module (e.g., `preProcessamento`, `io`, `models`).

3. **Run the Script**:
   - Adjust the number of folds (`k`, default = 5) and execute the script.

---

#### Output Files

- **Preprocessed Datasets**:
  - `classe0.csv` and `classe1.csv`.

- **Error Tracking Files**:
  - `<ModelName>Erros.csv`.

- **k-Fold Datasets**:
  - Training (`<prefix>Treino.csv`) and testing (`<prefix>Teste.csv`) datasets for each fold.

---

### For Linear Models (Logistic Regression and SGD)
#### Main File
- **`LinearModelsSpuriousTracker.py`**: Controls the pipeline from training to spurious pattern analysis.

#### Utility Modules (in `utils` folder)
- **`logistic_regression.py`**: Implements Logistic Regression with TF-IDF for pattern analysis.
- **`Method.py`**: Centralizes key functions, including:
  - Sentence explanations using LIME and Logistic Regression.
  - Error logging and similar sentence identification.
  - Pattern relevance calculations and categorization.
- **`models.py`**: Handles text vectorization, model training, evaluation metrics, and impact analysis.
- **`preProcessamento.py`**: Performs text cleaning and normalization.
- **`utils.py`**: Provides auxiliary functions for calculations, data manipulation, and file handling.
- **`importantFeatures.py`**: Identifies influential words in classification errors.
- **`ImportantFeaturesErrors.py`**: Analyzes sentences and patterns associated with errors.
- **`LimeUtils.py`**: Functions for local instance explanation using LIME.

### For BERT Model
#### Main File
- **`BERTSpuriousTracker.py`**: Orchestrates all pipeline stages.

#### Utility Modules (`/utils` folder)
- **`BERTcreateTrainTest.py`**: Creates training and testing datasets.
- **`BERTtrain.py`**: Trains the BERT model and calculates metrics.
- **`BERTexplainLimeGlobal.py`**: Generates global explanations using LIME.
- **`BERTgenerateIF.py`**: Processes and normalizes feature importance.
- **`BERTdisturbTest.py`**: Perturbs test sets for evaluation.
- **`BERTcalculateConf.py`**: Identifies spurious patterns.
- **`BERTcalculatePatterns.py`**: Extracts relevant patterns for analysis.
- **`BERTgerarResultados.py`**: Compiles final results.

---

## Clustering Using K-Means

### Overview
The `Clustering-K-Means` method identifies and analyzes confounding patterns in textual datasets using:
- WCSS for determining the optimal number of clusters.
- PCA for dimensionality reduction and visualization.
- 3D and 2D visualizations for cluster interpretation and pattern selection.

#### Main Script
- **`Clustering-K-Means.py`**: Executes the clustering pipeline.

#### Utility Modules
- **`utils/KMeans.py`**: Contains K-Means, PCA, and visualization utilities:
  - WCSS calculation.
  - Optimal cluster determination.
  - PCA-based heatmaps and visualizations.
  - Distance calculations to centroids and pattern selection.

---

## Clustering Results Summary
1. **Input Preparation**: Files should follow the naming convention `<Model>-<Dataset>-<Class>.csv` (e.g., `LRG-Contracts-0.csv`).
2. **Normalization**: Numeric columns are standardized using `StandardScaler`.
3. **WCSS Calculation**: Evaluates clustering performance.
4. **Optimal Clusters**: Determined using the Elbow Method.
5. **PCA Transformation**: Reduces dimensions to PC1, PC2, and PC3; heatmaps show variable contributions.
6. **KMeans Clustering**: Applies clustering on PCA results and extracts centroids.
7. **3D Visualization**: Highlights clusters, centroids, and pattern categories (closest, middle, furthest).
8. **2D/1D Visualizations**: Shows clusters and distance-based pattern spuriousness.

---

## Usage

### Prepare Input Files
Rename `ErrorsInfluence` files according to:
- Model: `LRG`
- Dataset: `Contracts`
- Class: `0`
- Example: `LRG-Contracts-0.csv`

---

## Usage
- warnings and logging are part of Python's standard library and do not require installation via pip.
- Ensure the correct Python version (3.7 or higher is recommended) is used to avoid compatibility issues, particularly with torch and transformers.
- To install all dependencies, run:
	pip install -r requirements.txt


## Datasets Used

This study utilized two datasets specifically developed in the context of the **Court of Auditors of the State of Piauí (TCE-PI)**:

### 1. Contracts Dataset (C)
The Contracts Dataset focuses on classifying contract descriptions related to public administration expenditures, particularly those linked to COVID-19.  
- **Development**: Initially labeled by a team of 12 TCE-PI specialists, the dataset includes descriptions of contract objects. Disagreements during the labeling process were resolved by a chief auditor.  
- **Classes**: 
  - **Label 1**: Healthcare-specific acquisitions.  
  - **Label 0**: Other procurements.  
- **Expansion**: The dataset was expanded using data from the Sistema Contratos - Web, adding 1,727 new sentences, resulting in 6,092 total descriptions.

### 2. Bidding Dataset (L)
The Bidding Dataset supports the detection of fraud indicators in public procurement.  
- **Development**: Specialists manually labeled procurement notices from 2012 to 2023 using categories such as services and goods acquisitions.  
- **Classes Adaptation**:
  - **Label 1**: Acquisition of goods (permanent and consumable).  
  - **Label 0**: Contracting of services (engineering and general).  
- **Expansion**: Through active learning cycles, the dataset was expanded from 2,137 to 6,244 sentences.

---

### Data Access
The datasets are **not publicly available** due to TCE-PI's internal regulations. However, the data can be provided upon request to the corresponding author, who is a representative of the institution. Requests must comply with TCE-PI's procedural formality.
