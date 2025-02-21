import argparse
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')

def tfidf_ngram(data, ngram_range=(1, 1)):
    """
    Trains and evaluates a logistic regression classifier using TF-IDF n-gram features (n-gram range specified by user).
    i.e., unigrams, bigrams, and trigrams = (1, 3)
    Default is unigrams (1, 1).

    Input: data (pd.DataFrame)
    Returns: df (pd.DataFrame) with the average accuracy, precision, recall, and F1 score.
    """
    tfidf_vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, token_pattern=None, ngram_range=ngram_range)
    
    # use entire dataset
    tfidf_features = tfidf_vectorizer.fit_transform(data['text'])

    clf_tfidf = LogisticRegression()
    clf_tfidf.fit(tfidf_features, data['intent'])
    
    return clf_tfidf, tfidf_vectorizer

def main():
    # data = pd.read_csv('diplomacy_cv.csv')
    # data2 = pd.read_csv('diplomacy_kaggle.csv')

    # clf_tfidf, vectorizer_tfidf = tfidf_ngram(data, ngram_range=(1, 1))
    # tfidf_test_features = vectorizer_tfidf.transform(data2['text'])
    # data2['intent'] = clf_tfidf.predict(tfidf_test_features)

    # submission = data2[['id', 'intent']]
    # submission.to_csv("submission.csv", index=False)

    parser = argparse.ArgumentParser(description="Train a deception classifier and make predictions.")
    parser.add_argument("--train", type=str, required=True, help="Path to the training dataset (CSV).")
    parser.add_argument("--test", type=str, required=True, help="Path to the test dataset (CSV).")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output CSV file for predictions.")
    
    args = parser.parse_args()

    print("\n=============================================")
    print(f"Loading training dataset: {args.train}")
    print(f"Loading test dataset: {args.test}")
    print(f"Output file: {args.output}")
    print("=============================================")

    train_data = pd.read_csv(args.train)
    test_data = pd.read_csv(args.test)

    clf_tfidf, vectorizer_tfidf = tfidf_ngram(train_data, ngram_range=(1, 1))

    tfidf_test_features = vectorizer_tfidf.transform(test_data['text'])
    test_data['intent'] = clf_tfidf.predict(tfidf_test_features)

    submission = test_data[['id', 'intent']]
    submission.to_csv(args.output, index=False)

    print("\n=============================================")
    print(f"Predictions saved to {args.output}")
    print("=============================================")

if __name__ == "__main__":
    main()