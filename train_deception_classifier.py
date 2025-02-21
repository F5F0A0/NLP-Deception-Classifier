import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report # this provides a bunch of useful evaluation metrics
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.tokenize import word_tokenize
from nltk.lm import Lidstone
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
nltk.download('vader_lexicon')
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
nltk.download('punkt')

parser = argparse.ArgumentParser(description="Run deception classifier on a dataset.")
parser.add_argument("--datafile", type=str, required=True, help="Path to the dataset CSV file.")

args = parser.parse_args()
datafile = args.datafile

print("\n=============================================")
print(f"Loading dataset: {datafile}")
print("=============================================")

data = pd.read_csv(datafile)

def nltk_tokenizer(text):
    return nltk.word_tokenize(text)

def bag_of_words_unigram(data):
    """
    Trains and evaluates a logistic regression classifier using bag-of-words (unigram) features.
    Performs 5-fold cross-validation to score performance.

    Input: data (pd.DataFrame)
    Returns: df (pd.DataFrame) with the average accuracy, precision, recall, and F1 score.
    """
    test_size = int(0.1 * len(data))
    train, test = train_test_split(data, test_size=test_size, random_state=6)

    unigram_vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize, token_pattern=None)
    unigram_vectorizer.fit(train['text'])

    unigram_train_features = unigram_vectorizer.transform(train['text'])
    unigram_test_features = unigram_vectorizer.transform(test['text'])

    clf_unigrams = LogisticRegression()
    clf_unigrams.fit(unigram_train_features, train['intent'])
    
    unigram_test_predictions = clf_unigrams.predict(unigram_test_features)
    test_labels = test['intent']
    
    print("\n=============================================")
    print("(Unigram) Classification Report (Test Set)")
    print("=============================================")
    results = pd.DataFrame(classification_report(test_labels, unigram_test_predictions, output_dict=True))
    print(results)

    print("\n=============================================")
    print("(Unigram) 5-Fold Cross-Validation Results")
    print("=============================================")
    pipeline = make_pipeline(CountVectorizer(tokenizer=nltk.word_tokenize, token_pattern=None), LogisticRegression())
    scores = cross_validate(pipeline, data['text'], data['intent'], cv=5, scoring=['accuracy', 'precision', 'recall', 'f1'], return_train_score=False)
    
    scores = pd.DataFrame(scores)[['test_accuracy', 'test_precision', 'test_recall', 'test_f1']]    
    print(scores.mean())

    return results, scores, clf_unigrams, unigram_vectorizer

def tfidf_ngram(data, ngram_range=(1, 2)):
    """
    Trains and evaluates a logistic regression classifier using TF-IDF unigram features.
    Performs 5-fold cross-validation to score performance.

    Input: data (pd.DataFrame)
    Returns: df (pd.DataFrame) with the average accuracy, precision, recall, and F1 score.
    """
    test_size = int(0.1 * len(data))
    train, test = train_test_split(data, test_size=test_size, random_state=6)

    tfidf_vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, token_pattern=None, ngram_range=ngram_range)
    tfidf_vectorizer.fit(train['text'])

    tfidf_train_features = tfidf_vectorizer.transform(train['text'])
    tfidf_test_features = tfidf_vectorizer.transform(test['text'])

    clf_tfidf = LogisticRegression()
    clf_tfidf.fit(tfidf_train_features, train['intent'])
    
    tfidf_test_predictions = clf_tfidf.predict(tfidf_test_features)
    test_labels = test['intent']
    
    print("\n=============================================")
    print(f"(TF-IDF {ngram_range[0]}-{ngram_range[1]}gram) Classification Report (Test Set)")
    print("=============================================")
    results = pd.DataFrame(classification_report(test_labels, tfidf_test_predictions, output_dict=True))
    print(results)

    print("\n=============================================")
    print(f"(TF-IDF {ngram_range[0]}-{ngram_range[1]}gram) 5-Fold Cross-Validation Results")
    print("=============================================")
    pipeline = make_pipeline(TfidfVectorizer(tokenizer=nltk.word_tokenize, token_pattern=None, ngram_range=ngram_range), LogisticRegression())
    scores = cross_validate(pipeline, data['text'], data['intent'], cv=5, scoring=['accuracy', 'precision', 'recall', 'f1'], return_train_score=False)
    
    scores = pd.DataFrame(scores)[['test_accuracy', 'test_precision', 'test_recall', 'test_f1']]    
    print(scores.mean())

    return results, scores, clf_tfidf, tfidf_vectorizer

def compute_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment['compound'], sentiment['pos'], sentiment['neg'], sentiment['neu']

class SentimentTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer to compute sentiment features using VADER.
    """
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def fit(self, X, y=None):
        return self  # no fitting required

    def transform(self, X):
        sentiment_scores = [self.sia.polarity_scores(text) for text in X]
        sentiment_df = pd.DataFrame(sentiment_scores)
        return sentiment_df[['compound', 'pos', 'neg', 'neu']].values

def tfidf_ngram_sentiment(data, ngram_range=(1, 1)):
    """
    Trains and evaluates a logistic regression classifier using TF-IDF n-gram features + Sentiment Features.
    Performs 5-fold cross-validation to score performance.

    Input: data (pd.DataFrame)
    Returns: results, scores, trained model, trained vectorizer
    """
    test_size = int(0.1 * len(data))
    train, test = train_test_split(data, test_size=test_size, random_state=6)

    train_sentiments = train['text'].apply(compute_sentiment).apply(pd.Series)
    test_sentiments = test['text'].apply(compute_sentiment).apply(pd.Series)
    train_sentiments.columns = ['compound', 'positive', 'negative', 'neutral']
    test_sentiments.columns = ['compound', 'positive', 'negative', 'neutral']

    scaler = StandardScaler(with_mean=False)
    train_sentiments = scaler.fit_transform(train_sentiments)
    test_sentiments = scaler.transform(test_sentiments)

    tfidf_vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, token_pattern=None, ngram_range=ngram_range)
    tfidf_vectorizer.fit(train['text'])

    tfidf_train_features = tfidf_vectorizer.transform(train['text'])
    tfidf_test_features = tfidf_vectorizer.transform(test['text'])

    train_features = hstack([tfidf_train_features, train_sentiments])
    test_features = hstack([tfidf_test_features, test_sentiments])

    clf_tfidf_sentiment = LogisticRegression()
    clf_tfidf_sentiment.fit(train_features, train['intent'])

    tfidf_sentiment_test_predictions = clf_tfidf_sentiment.predict(test_features)
    test_labels = test['intent']

    print("\n=============================================")
    print(f"(TF-IDF {ngram_range[0]}-{ngram_range[1]}gram + Sentiment Features) Classification Report (Test Set)")
    print("=============================================")
    results = pd.DataFrame(classification_report(test_labels, tfidf_sentiment_test_predictions, output_dict=True))
    print(results)

    pipeline = make_pipeline(
        FeatureUnion([
            ("tfidf", TfidfVectorizer(tokenizer=nltk.word_tokenize, token_pattern=None, ngram_range=ngram_range)),
            ("sentiment", SentimentTransformer())
        ]),
        StandardScaler(with_mean=False),
        LogisticRegression()
    )

    scores = cross_validate(
        pipeline,
        data['text'], data['intent'], 
        cv=5, scoring=['accuracy', 'precision', 'recall', 'f1'],
        return_train_score=False
    )

    scores = pd.DataFrame(scores)[['test_accuracy', 'test_precision', 'test_recall', 'test_f1']]
    
    print("\n=============================================")
    print(f"(TF-IDF {ngram_range[0]}-{ngram_range[1]}gram + Sentiment Features) 5-Fold Cross-Validation Results")
    print("=============================================")
    print(scores.mean())

    return results, scores, clf_tfidf_sentiment, tfidf_vectorizer

def most_informative_features(feature_names, classifier, class_id=1, n=20):
    assert len(feature_names) == classifier.coef_.shape[1]
    if class_id == 1: # positive class
        topn = reversed(sorted(zip(classifier.coef_[0], feature_names))[-n:])
    else: # negative class
        topn = sorted(zip(classifier.coef_[0], feature_names))[:n]
    for coef, feat in topn:
        print(feat, coef)

def compute_confusion_matrix(test_labels, test_predictions):
    cm = confusion_matrix(test_labels, test_predictions)
    cm_df = pd.DataFrame(cm, index=["Truthful (0)", "Lying (1)"], columns=["Predicted Truthful (0)", "Predicted Lying (1)"])
    
    print("\n=============================================")
    print("Confusion Matrix")
    print("=============================================")
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.show()

    return cm_df

def sample_errors(data, test_labels, test_predictions):
    # identify false positives and false negatives
    data['predicted'] = test_predictions  # add predictions to test dataframe

    false_positives = data[(test_labels == 0) & (test_predictions == 1)]  # truthful predicted as lying
    false_negatives = data[(test_labels == 1) & (test_predictions == 0)]  # lies predicted as truthful

    print("\nFalse Positives (Predicted as Lying, but was Truthful):")
    print(false_positives[['text']].sample(min(3, len(false_positives)), random_state=42))

    print("\nFalse Negatives (Predicted as Truthful, but was Lying):")
    print(false_negatives[['text']].sample(min(3, len(false_negatives)), random_state=42))

def error_analysis(model, vectorizer, test_data):
    print("\n=============================================")
    print("Error Analysis")
    print("=============================================")
    
    X_test = vectorizer.transform(test_data['text'])
    y_test = test_data['intent']
    
    y_pred = model.predict(X_test)
    
    confusion = pd.DataFrame(confusion_matrix(y_test, y_pred), 
                             columns=['Predicted Truthful (0)', 'Predicted Lying (1)'], 
                             index=['Actual Truthful (0)', 'Actual Lying (1)'])
    print("\n=============================================")
    print("Confusion Matrix")
    print("=============================================")
    print(confusion)

    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion, annot=True, fmt='d', cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.show()

    test_data['prediction'] = y_pred

    false_positives = test_data[(y_test == 0) & (y_pred == 1)]
    false_negatives = test_data[(y_test == 1) & (y_pred == 0)]

    print("\n=============================================")
    print("Sample False Positives (Predicted as Lying, Actually Truthful)")
    print("=============================================")
    pd.set_option('display.max_colwidth', None)
    print(false_positives[['text']].sample(min(5, len(false_positives)), random_state=6))

    print("\n=============================================")
    print("Sample False Negatives (Predicted as Truthful, Actually Lying)")
    print("=============================================")
    pd.set_option('display.max_colwidth', None)
    print(false_negatives[['text']].sample(min(5, len(false_negatives)), random_state=6))

def main():
    # take the filename of a dataset as a single keyword argument
    # data = pd.read_csv('diplomacy_cv.csv')

    unigram_results, unigram_scores, clf_unigram, vectorizer_unigram = bag_of_words_unigram(data)

    tfidf_unigram_results, tfidf_unigram_scores, clf_tfidf_1, vectorizer_tfidf_1 = tfidf_ngram(data, ngram_range=(1, 1))
    tfidf_bigram_results, tfidf_bigram_scores, clf_tfidf_2, vectorizer_tfidf_2 = tfidf_ngram(data, ngram_range=(1, 2))
    tfidf_trigram_results, tfidf_trigram_scores, clf_tfidf_3, vectorizer_tfidf_3 = tfidf_ngram(data, ngram_range=(1, 3))
    tfidf_4gram_results, tfidf_4gram_scores, clf_tfidf_4, vectorizer_tfidf_4 = tfidf_ngram(data, ngram_range=(1, 4))

    tfidf_sentiment_unigram_results, tfidf_sentiment_unigram_scores, clf_tfidf_sent_1, vectorizer_tfidf_sent_1 = tfidf_ngram_sentiment(data, ngram_range=(1, 1))
    tfidf_sentiment_bigram_results, tfidf_sentiment_bigram_scores, clf_tfidf_sent_2, vectorizer_tfidf_sent_2 = tfidf_ngram_sentiment(data, ngram_range=(1, 2))
    tfidf_sentiment_trigram_results, tfidf_sentiment_trigram_scores, clf_tfidf_sent_3, vectorizer_tfidf_sent_3 = tfidf_ngram_sentiment(data, ngram_range=(1, 3))
    tfidf_sentiment_4gram_results, tfidf_sentiment_4gram_scores, clf_tfidf_sent_4, vectorizer_tfidf_sent_4 = tfidf_ngram_sentiment(data, ngram_range=(1, 4))

    results_df = pd.DataFrame({
        "Model": [
            "Unigram (Bag-of-Words)",
            "TF-IDF Unigrams",
            "TF-IDF Bigrams",
            "TF-IDF Trigrams",
            "TF-IDF 4-grams",
            "TF-IDF Unigrams + Sentiment",
            "TF-IDF Bigrams + Sentiment",
            "TF-IDF Trigrams + Sentiment",
            "TF-IDF 4-grams + Sentiment"
        ],
        "Accuracy": [
            unigram_scores["test_accuracy"].mean(),
            tfidf_unigram_scores["test_accuracy"].mean(),
            tfidf_bigram_scores["test_accuracy"].mean(),
            tfidf_trigram_scores["test_accuracy"].mean(),
            tfidf_4gram_scores["test_accuracy"].mean(),
            tfidf_sentiment_unigram_scores["test_accuracy"].mean(),
            tfidf_sentiment_bigram_scores["test_accuracy"].mean(),
            tfidf_sentiment_trigram_scores["test_accuracy"].mean(),
            tfidf_sentiment_4gram_scores["test_accuracy"].mean()
        ],
        "Precision (Lying)": [
            unigram_scores["test_precision"].mean(),
            tfidf_unigram_scores["test_precision"].mean(),
            tfidf_bigram_scores["test_precision"].mean(),
            tfidf_trigram_scores["test_precision"].mean(),
            tfidf_4gram_scores["test_precision"].mean(),
            tfidf_sentiment_unigram_scores["test_precision"].mean(),
            tfidf_sentiment_bigram_scores["test_precision"].mean(),
            tfidf_sentiment_trigram_scores["test_precision"].mean(),
            tfidf_sentiment_4gram_scores["test_precision"].mean()
        ],
        "Recall (Lying)": [
            unigram_scores["test_recall"].mean(),
            tfidf_unigram_scores["test_recall"].mean(),
            tfidf_bigram_scores["test_recall"].mean(),
            tfidf_trigram_scores["test_recall"].mean(),
            tfidf_4gram_scores["test_recall"].mean(),
            tfidf_sentiment_unigram_scores["test_recall"].mean(),
            tfidf_sentiment_bigram_scores["test_recall"].mean(),
            tfidf_sentiment_trigram_scores["test_recall"].mean(),
            tfidf_sentiment_4gram_scores["test_recall"].mean()
        ],
        "F1 Score (Lying)": [
            unigram_scores["test_f1"].mean(),
            tfidf_unigram_scores["test_f1"].mean(),
            tfidf_bigram_scores["test_f1"].mean(),
            tfidf_trigram_scores["test_f1"].mean(),
            tfidf_4gram_scores["test_f1"].mean(),
            tfidf_sentiment_unigram_scores["test_f1"].mean(),
            tfidf_sentiment_bigram_scores["test_f1"].mean(),
            tfidf_sentiment_trigram_scores["test_f1"].mean(),
            tfidf_sentiment_4gram_scores["test_f1"].mean()
        ]
    })

    print("\n=============================================")
    print("5-Fold Cross-Validation Performance Table")
    print("=============================================")
    print(results_df)

    feature_names_trigram = vectorizer_tfidf_3.get_feature_names_out()
    print("\n=============================================")
    print("Most Informative Features for Lying")
    print("=============================================")
    most_informative_features(feature_names_trigram, clf_tfidf_3, class_id=1, n=20)
    print("\n=============================================")
    print("Most Informative Features for Truthfulness:")
    print("=============================================")
    most_informative_features(feature_names_trigram, clf_tfidf_3, class_id=0, n=20)

    print("\n=============================================")
    print("Error Analysis on TF-IDF Trigram Model:")
    print("=============================================")
    error_analysis(clf_tfidf_3, vectorizer_tfidf_3, data)

if __name__ == "__main__":
    main()