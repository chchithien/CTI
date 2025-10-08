import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv("Desktop\\CTI\\dataSet2\\emails_features.csv")

# 2. Features and target
X = df.drop(columns=["Spam/Ham"])
y = df["Spam/Ham"]

# 3. Split train & test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Random Forest does not require feature scaling (tree models are scale-invariant)
X_train_scaled, X_test_scaled = X_train, X_test

# 4. Random Forest + GridSearchCV
param_grid = {
    "n_estimators": [100, 200],       # Number of trees
    "max_depth": [None, 10, 20],      # Maximum tree depth
    "min_samples_split": [2, 5, 10],  # Minimum samples to split a node
    "min_samples_leaf": [1, 2, 4],    # Minimum samples at leaf node
    "class_weight": [None, "balanced"] # Handle class imbalance
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid,
    scoring="f1",  # Optimize for F1-score
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV F1 Score:", grid_search.best_score_)

# 5. Predict with the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# 6. Model evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nRandom Forest (Best Model) Results:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

# 7. Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest (Best Params)")
plt.show()
