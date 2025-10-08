# Enron Spam Email Dataset

## 1.Dataset Overview
This project contains two related datasets used for spam email classification.

- **`enron_spam_data.csv`** - Raw Dataset  
  Contains unprocessed email text and spam/ham labels collected from the Enron email corpus.

- **`emails_clean.csv`** - Clean Dataset  
  The processed version of the raw dataset. Text has been cleaned by removing HTML tags, punctuation, stop words, and converting to lowercase. This dataset is ready for feature extraction and model training.

---

## 2.Dataset Purpose
Both datasets are designed for **spam email detection**, where the goal is to classify emails as either *spam* or *ham (non-spam)* based on their content.

---

## 3.Feature Information
### Raw Dataset (`enron_spam_data.csv`)
| Column | Description |
|--------|--------------|
| `text` | The full, unprocessed email content |
| `label` | Indicates whether the email is spam or ham |

### Clean Dataset (`emails_clean.csv`)
| Column | Description |
|--------|--------------|
| `ID` |  |
| `clean_subject` | Preprocessed text used for TF-IDF vectorization |
| `clean_text` | Preprocessed text used for TF-IDF vectorization |
| `label` | Indicates whether the email is spam or ham |

---

## 44.Relationship between Datasets
The **clean dataset** (`emails_clean.csv`) was created by cleaning and transforming the **raw dataset** (`enron_spam_data.csv`).  
This ensures consistent and noise-free input for feature extraction and machine learning model training.

---

## 5.Usage
- Use `enron_spam_data.csv` if you want to perform your own text preprocessing.
- Use `emails_clean.csv` directly for model training and evaluation.

---

