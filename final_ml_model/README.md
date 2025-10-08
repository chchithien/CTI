# Spam Email Detection using Machine Learning

A machine learning project that compares **Naive Bayes**, **Random Forest**, and **Logistic Regression** models to detect spam emails with high accuracy.

---

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Format](#dataset-format)
- [How to Run](#how-to-run)
- [Output](#output)
- [Project Structure](#project-structure)
- [Results](#results)

---

## Features

- **Data Preprocessing**: Text cleaning, duplicate removal, outlier handling
- **Feature Extraction**: TF-IDF vectorization + numeric features
- **Model Comparison**: 3 ML models compared side-by-side
- **Performance Visualization**: ROC and Precision-Recall curves
- **Best Model Selection**: Automatically picks best performing model
- **Detailed Metrics**: Accuracy, Precision, Recall, F1-Score

---

## Requirements

### Software
- Python 3.7+
- pip (Python package manager)

### Python Libraries
```bash
pandas
numpy
scikit-learn
matplotlib
scipy
```

---

## Installation

### Step 1: Clone or Download the Project
```bash
# Clone the repository (if using git)
git clone <your-repo-url>
cd spam-detection

# OR download and extract the ZIP file
```

### Step 2: Install Required Libraries
```bash
# Install all required packages
pip install pandas numpy scikit-learn matplotlib scipy

# OR install from requirements.txt (if provided)
pip install -r requirements.txt
```

### Step 3: Prepare Your Dataset
Place your spam email dataset (CSV file) in the project folder and name it:
```
spam_emails.csv
```

---

## Dataset Format

Your CSV file should have **at least 2 columns**:

### Option 1: Standard Format
| v1   | v2                                    |
|------|---------------------------------------|
| ham  | Go until jurong point, crazy..        |
| spam | Free entry in 2 a wkly comp to win... |
| ham  | U dun say so early hor...             |

### Option 2: Alternative Format
| label | text                                  |
|-------|---------------------------------------|
| ham   | Go until jurong point, crazy..        |
| spam  | Free entry in 2 a wkly comp to win... |

**Note**: The code automatically detects your column names!

---

## How to Run

### Method 1: Run Complete Pipeline (Recommended)
This runs everything: preprocessing → cleaning → training → results

```bash
python simple_spam_pipeline.py
```

**OR** if using the notebook:
```bash
python ML_Model_CTI.py
```

### Method 2: Run Step-by-Step

#### Step 1: Preprocess & Clean Data
```bash
python simple_spam_pipeline.py preprocess
```
**Output**: Creates `features.pkl`, `labels.csv`, `tfidf_vectoriser.pkl`, `scaler.pkl`

#### Step 2: Train Models
```bash
python simple_spam_pipeline.py train
```
**Output**: Shows model comparison and performance curves

### Method 3: Use Google Colab

1. Upload `ML_Model_CTI.ipynb` to Google Colab
2. Upload your `spam_emails.csv` when prompted
3. Run all cells (Runtime → Run all)
4. View results and download charts

---

## Output

### Console Output Example:
```
STEP 1: PREPROCESSING & CLEANING
==================================================
Loading data...
Original shape: (5572, 2)
Removed 45 duplicates
Final shape: (5234, 2)

STEP 2: VECTORISING & SAVING
==================================================
Vectorising features...
Features shape: (5234, 3003)
Saved: features.pkl, labels.csv, tfidf_vectoriser.pkl, scaler.pkl

STEP 3: TRAINING MODELS
==================================================
Train: (4187, 3003), Test: (1047, 3003)

=== Naive Bayes ===
Accuracy : 0.9542
Precision: 0.9823
Recall   : 0.9156
F1-score : 0.9478

=== Random Forest ===
Accuracy : 0.9687
Precision: 0.9765
Recall   : 0.9534
F1-score : 0.9648

=== Logistic Regression ===
Accuracy : 0.9721
Precision: 0.9812
Recall   : 0.9589
F1-score : 0.9699

======================================================================
MODEL COMPARISON SUMMARY
======================================================================
Model                Accuracy     Precision    Recall       F1-Score    
----------------------------------------------------------------------
Naive Bayes          0.9542       0.9823       0.9156       0.9478      
Random Forest        0.9687       0.9765       0.9534       0.9648      
Logistic Regression  0.9721       0.9812       0.9589       0.9699      

BEST MODEL: Logistic Regression (F1-Score: 0.9699)

ROC-AUC Score: 0.9840
Average Precision Score: 0.9630
```

### Generated Files:
- `features.pkl` - Vectorized features
- `labels.csv` - Email labels
- `tfidf_vectoriser.pkl` - TF-IDF model
- `scaler.pkl` - Feature scaler
- `model_performance_curves.png` - ROC & PR curves

---

## Project Structure

```
spam-detection/
│
├── spam_emails.csv              # Your input dataset
├── simple_spam_pipeline.py      # Main Python script
├── ML_Model_CTI.ipynb          # Jupyter/Colab notebook
├── README.md                    # This file
│
├── features.pkl                 # Generated: Vectorized features
├── labels.csv                   # Generated: Labels
├── tfidf_vectoriser.pkl        # Generated: TF-IDF model
├── scaler.pkl                   # Generated: Scaler
└── model_performance_curves.png # Generated: Performance charts
```

---

## Results

### Expected Performance:

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Naive Bayes         | ~95%     | ~98%      | ~92%   | ~95%     |
| Random Forest       | ~97%     | ~98%      | ~95%   | ~96%     |
| Logistic Regression | ~97%     | ~98%      | ~96%   | ~97%     |

**Winner**: Logistic Regression typically performs best!

### What the Metrics Mean:
- **Accuracy**: Overall correctness (96-97% correct predictions)
- **Precision**: When it says "spam", how often it's right (98%)
- **Recall**: How many spam emails it catches (96%)
- **F1-Score**: Balanced measure of precision & recall (97%)

---

## Troubleshooting

### Error: "No module named 'sklearn'"
```bash
pip install scikit-learn
```

### Error: "File not found: spam_emails.csv"
- Make sure your CSV file is in the same folder as the script
- Check the filename is exactly: `spam_emails.csv`

### Error: "KeyError: 'v1' or 'v2'"
- Your dataset has different column names
- The code will auto-detect most formats, but check your CSV structure

### Memory Error
- Your dataset might be too large
- Try reducing `max_features` in the code from 3000 to 1000

---

## Notes

- **Training Time**: 1-3 minutes on average datasets (5000-10000 emails)
- **Minimum Dataset Size**: At least 1000 emails recommended
- **Best Model**: Usually Logistic Regression or Random Forest
- **Saved Models**: Can be reused for future predictions
