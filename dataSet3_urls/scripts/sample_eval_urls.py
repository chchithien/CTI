
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(Path(__file__).resolve().parents[1] / "processed" / "dataset3_urls.csv")
X = df.drop(columns=["url","label"])
y = df["label"].astype(int)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler(with_mean=False)
Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
lr = LogisticRegression(max_iter=1000)
lr.fit(Xtr_s, ytr)
pred_lr = lr.predict(Xte_s)
print("=== Logistic Regression ===")
print(classification_report(yte, pred_lr, digits=4))

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(Xtr, ytr)
pred_rf = rf.predict(Xte)
print("\\n=== RandomForest ===")
print(classification_report(yte, pred_rf, digits=4))

kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
kmeans.fit(X)
df["cluster"] = kmeans.labels_
print("\\n=== Clustering alignment (label vs. cluster) ===")
print(pd.crosstab(df["label"], df["cluster"], margins=True))
