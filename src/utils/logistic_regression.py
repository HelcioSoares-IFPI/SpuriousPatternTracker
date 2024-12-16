from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def train_explaner(X_train, y_train):
    """
    Train a logistic regression model with TF-IDF vectorization.

    Args:
        X_train (list): List of training texts.
        y_train (list): List of training labels.

    Returns:
        tuple: Trained model and TF-IDF vectorizer.
    """
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, vectorizer

def evaluate_explaner(vectorizer, model, X_test, y_test):
    """
    Evaluate the model using the test data and return a classification report.

    Args:
        vectorizer (TfidfVectorizer): TF-IDF vectorizer.
        model (LogisticRegression): Trained logistic regression model.
        X_test (list): List of test texts.
        y_test (list): List of test labels.

    Returns:
        str: Classification report.
    """
    X_test = vectorizer.transform(X_test)
    prediction = model.predict(X_test)
    report = classification_report(y_test, prediction)
    return report

def get_score(vectorizer, model, text):
    """
    Get the score of each word in the text based on the model's coefficients.

    Args:
        vectorizer (TfidfVectorizer): TF-IDF vectorizer.
        model (LogisticRegression): Trained logistic regression model.
        text (str): Text to analyze.

    Returns:
        list: Two dictionaries containing the scores of words for each class.
    """
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_.flatten()
    s_vectorized = vectorizer.transform([text])
    s_features = s_vectorized.nonzero()[1]
    class_0, class_1 = [], []
    for feature_name, coef in zip([feature_names[i] for i in s_features], [coefficients[i] for i in s_features]):
        if coef > 0:
            class_1.append((feature_name, coef))
        else:
            class_0.append((feature_name, abs(coef)))
    class_0 = sorted(class_0, key=lambda x: x[1], reverse=True)
    class_1 = sorted(class_1, key=lambda x: x[1], reverse=True)

    return [{feature: coef for feature, coef in class_0}, {feature: coef for feature, coef in class_1}]
