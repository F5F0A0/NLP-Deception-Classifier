# NLP-Deception-Classifier
NLP Deception Classifier using logistic regression with TF-IDF, n-grams, and sentiment analysis. Evaluates performance via 5-fold cross-validation and confusion matrix analysis. 
Achieved 67% accuracy on a hidden Kaggle dataset. 
Developed as part of CS 1671: Human Language Technologies. 
Data sourced from the Diplomacy deception corpus (Peskov et al., 2020).


install requirements.txt
python3 train_deception_classifier.py --datafile diplomacy_cv.csv
python3 test_deception_classifier.py --train diplomacy_cv.csv --test diplomacy_kaggle.csv --output submission.csv
