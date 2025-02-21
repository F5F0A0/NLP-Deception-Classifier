# NLP-Deception-Classifier
A deception classifier using **logistic regression** with **TF-IDF, n-grams, and sentiment analysis**.  
Evaluates performance via **5-fold cross-validation** and **confusion matrix analysis**.  
Achieved **67% accuracy** on a hidden Kaggle dataset.

## Usage
```bash
pip install -r requirements.txt (or conda install --file requirements.txt)
python3 train_deception_classifier.py --datafile diplomacy_cv.csv  
python3 test_deception_classifier.py --train diplomacy_cv.csv --test diplomacy_kaggle.csv --output submission.csv  
```

## Author
**Bridget Brinkman**  
GitHub: [@F5F0A0](https://github.com/F5F0A0)  
Project for **CS 1671: Human Language Technologies, Spring 2025.**  
Data sourced from the Diplomacy deception corpus (Peskov et al., 2020).
