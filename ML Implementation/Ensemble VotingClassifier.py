import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv("Desktop\\CTI\\emails_features.csv")

# 2. Features and target
X = df.drop(columns=["Spam/Ham"])
y = df["Spam/Ham"]

# 3. Feature scaling (for LogisticRegression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Initialize models
lr = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
nb = GaussianNB()

# 5. Ensemble (Soft Voting)
ensemble = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('nb', nb)],
    voting='soft'
)

# 6. Train on full dataset
ensemble.fit(X_scaled, y)

# 7. Predict on full dataset (same data)
y_pred = ensemble.predict(X_scaled)

# 8. Evaluation
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

print("\n=== Ensemble Voting Classifier (Full Dataset) ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# 9. Confusion Matrix Visualization
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Ensemble (Full Dataset)")
plt.show()
