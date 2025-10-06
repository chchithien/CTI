
# Train/evaluate simple baselines on the base email dataset using TF-IDF.
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

DATA_DIR = Path("data/processed_base")
train_path = DATA_DIR / "base_emails_train.csv"
test_path = DATA_DIR / "base_emails_test.csv"

if not train_path.exists() or not test_path.exists():
    raise FileNotFoundError("Run scripts/preprocess_base_emails.py first to generate train/test splits.")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

Xtr, ytr = train["text_clean"].astype(str), train["label"].astype(int)
Xte, yte = test["text_clean"].astype(str), test["label"].astype(int)

# Model 1: Logistic Regression
lr = make_pipeline(TfidfVectorizer(max_features=30000, ngram_range=(1,2)), LogisticRegression(max_iter=200))
lr.fit(Xtr, ytr)
pred_lr = lr.predict(Xte)
print("=== Logistic Regression ===")
print(classification_report(yte, pred_lr, digits=4))
print(confusion_matrix(yte, pred_lr))

# Model 2: RandomForest (on TF-IDF)
rf = make_pipeline(TfidfVectorizer(max_features=20000), RandomForestClassifier(n_estimators=300, random_state=42))
rf.fit(Xtr, ytr)
pred_rf = rf.predict(Xte)
print("\n=== RandomForest ===")
print(classification_report(yte, pred_rf, digits=4))
print(confusion_matrix(yte, pred_rf))
