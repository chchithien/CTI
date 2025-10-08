#  Email Spam Detection Using Machine Learning

##  Overview
This project develops a **machine learning-based email spam classifier**
using data cleaning, feature extraction, and Machine learning implementation.  
It integrates multiple datasets (`emails.csv`, `enron_spam_data.csv`,
etc.), performs comprehensive preprocessing, and builds a trained model
capable of predicting whether an email is spam or ham (legitimate).

---

##  1. Environment Configuration

### Step 1. Create Conda Environment
```bash
conda env create -f environment.yml
conda activate spamdetect
```

---

##  2. Data Cleaning and Preprocessing

### 🔹 Dataset1 cleaning (`data_clean_dataset1.py`)
This script performs step-by-step preprocessing such as:  
- Removing stopwords, URLs, HTML tags, and special characters  
- Applying stemming or lemmatization  
- Detecting hotwords like "free", "cash", "winner", etc.  
- Extracting metadata features (emoji use, exclamation marks, email addresses, etc.)

**Run:**
```bash
python data_clean_dataset1.py
```
**Output file:**  
`preprocessed_spam.csv`

---

### 🔹 Dataset2 cleaning (`data_clean_dataset2.py`)
For dataset2 cleaning:  
- Removes headers, tables, dates, times, emails, and symbols  
- Filters outliers and extremely short messages  
- Converts "Spam/Ham" to binary (1 for spam, 0 for ham)  
- Outputs a refined dataset ready for feature extraction

**Run:**
```bash
python data_clean_dataset2.py
```
**Output file:**  
`emails_clean.csv`

---

### 🔹 Dataset3 cleaning (`data_cleaning_dataset3.py`)

**Main features:**  
- Uses a `SpamDataProcessor` class for modular data handling  
- Performs feature extraction (text length, word count, punctuation counts, etc.)  
- Removes duplicates and cleans numeric outliers using IQR  
- Detects URLs and email patterns  
- Outputs a summarized report and saves cleaned data  

**Run:**
```bash
python data_cleaning_dataset3.py
```
**Output file:**  
`cleaned_spam_data.csv`

This version is designed for large or mixed-format datasets and serves as an improved, reusable pipeline for future dataset preprocessing.

---

### 🔹 Combined Dataset
After cleaning, all processed datasets are **merged** into a unified file:

```bash
combined_dataset.csv
```

This file serves as the **main input dataset** for feature extraction and model training.  
It ensures consistency by combining cleaned samples from multiple sources (e.g., Enron and general spam email sets).

---

##  3. Feature Extraction

Use `feature_extraction.py` to convert cleaned text into structured numerical features.

**Features include:**  
- Structural statistics (message length, exclamation ratios, digit ratios, URL count)  
- Keyword presence (spam words like "money", "win", "offer")  
- TF-IDF text embeddings  

**Run:**
```bash
python feature_extraction.py
```

**Output file:**  
`emails_features.csv`  
(Feature matrix saved for model training)

---

##  4. Model Training (Random Forest) 
It was listed separately because other team members combined their methods into a single file.

Use `randomforest.py` to train and evaluate the spam classifier.

**Core Steps:**  
- Splits data into training (80%) and testing (20%) sets  
- Performs hyperparameter tuning with `GridSearchCV`  
- Evaluates accuracy, precision, recall, F1-score, and ROC-AUC  
- Visualizes the confusion matrix, ROC, and Precision-Recall curves  

**Run:**
```bash
python randomforest.py
```

**Results include:**  
- Best Random Forest hyperparameters  
- Model evaluation metrics printed in the terminal  
- Visualization plots for performance evaluation  

---

###  5.1 Integrated Notebook (`ml_model_cti.ipynb`)

The Jupyter Notebook **`ml_model_cti.ipynb`** provides an interactive environment to:  
- Combine data cleaning, feature extraction, and model training in one workflow  
- Run experiments step by step, visualizing accuracy, confusion matrix, and ROC curves  
- Quickly adjust hyperparameters and feature extraction settings  
- Export trained models and TF-IDF vectorizers for later use  

You can open it with Jupyter Lab or Notebook:
```bash
jupyter lab ml_model_cti.ipynb
```

This notebook is particularly useful for debugging, tuning, and presenting the project process interactively.

---

###  5.2 Models Used in `ml_model_cti.ipynb`

The notebook evaluates and compares several **machine learning algorithms** for spam detection.  
Each model is trained using the same preprocessed and feature-extracted dataset (`combined_dataset.csv` → `emails_features.csv`), and their results are compared.

**Models Implemented:**
1. **Random Forest Classifier**  — Main model, tuned with `GridSearchCV`  
2. **Logistic Regression**  — Baseline linear model  
3. **Naïve Bayes (MultinomialNB / BernoulliNB)**  — Fast, efficient, classic spam detector  
4. **Support Vector Machine (SVM)**  — Robust classifier for high-dimensional data  
5. **Decision Tree / KNN**  — Tested for comparison  

**Comparison Metrics:**  
Accuracy, Precision, Recall, F1-score, ROC-AUC, Execution time  

---

##  6. Files Summary

| File | Description |
|------|--------------|
| `emails.csv` / `enron_spam_data.csv` | Original raw email datasets |
| `data_clean_dataset1.py` | cleaned dataset |
| `data_clean_dataset2.py` | cleaned dataset |
| `data_cleaning_dataset3.py` | cleaned dataset |
| `combined_dataset.csv` | Unified dataset after merging all cleaned outputs |
| `feature_extraction.py` | TF-IDF and statistical feature extraction |
| `emails_features.csv` | Final feature matrix for model input |
| `randomforest.py` | Random Forest training, tuning, and visualization |
| `ml_model_cti.ipynb` | Jupyter Notebook integrating all ML models and experiments |

---

##  7. Expected Output Example
```text
Best Parameters: {'n_estimators': 200, 'max_depth': 20, 'class_weight': 'balanced'}
Best CV F1 Score: 0.974
Accuracy : 0.982
Precision: 0.977
Recall   : 0.971
F1-score : 0.974
ROC AUC: 0.991
```

---

##  8. Notes
- Ensure **NLTK datasets** (`punkt`, `wordnet`, `stopwords`) are downloaded automatically.  
- Update file paths (`Desktop\CTI\dataSet2\...`) according to your directory.  
- Each script produces `.csv` outputs for the next processing stage.  
