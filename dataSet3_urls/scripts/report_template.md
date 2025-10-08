
# Project Title: Spam & Phishing Detection (AI4Cyber)

## Group Identities
- Group: Session15-Group4
- Members: Chi Thien Ly (105223103), Jinxi Chen (105306677), Vallerian Raimon (105078112)
- Tutor: <Ricky Dong>

## 1. Introduction
- Problem motivation (email spam & phishing URLs)
- Intended users (e.g., students, general public, org security teams)

## 2. Problem Framing
- Why ML fits; cost of false negatives
- Task types used: **classification** + **clustering** (optionally regression)

## 3. Data Collection
- Base email dataset (school-provided) — date loaded: 2025-10-05
- URL dataset (Dataset 3): OpenPhish (malicious) + Tranco (benign). Collection date: <fill AEST>
- Rationale for benign sampling; balancing
- Challenges (duplicates, noisy text)

## 4. Data Processing
- Emails: lowercasing, remove URLs/emails/numbers/punct, collapse whitespace
- Deduplicate on cleaned text, drop empty
- Labels mapped to 0/1
- URL features: lexical (lengths, special chars, suspicious TLD, keyword flags)

## 5. Model Selection
- Baselines: Logistic Regression (interpretable), RandomForest (non-linear)
- Unsupervised: KMeans for structure discovery

## 6. Implementation
- Python, pandas, scikit-learn
- Repro steps with conda (see README)
- Notes on challenges (wrangling, class balance)

## 7. Evaluation
- Report Accuracy, Precision, Recall, F1 (macro + per-class)
- Emphasize Recall for positive (spam/phish)
- Clustering cross-tab vs labels

## 8. Conclusion
- Key findings & limitations
- Future work (content-based, transformers, calibration, web demo)

## 9. Bibliography (Harvard)
- OpenPhish community feed
- Tranco top sites
- scikit-learn docs, etc.

## Appendix
- Confusion matrices, feature importance plots (screenshots readable at A4)
