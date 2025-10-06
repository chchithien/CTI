# AI4Cyber — Assignment 2 (Spam/Malware Detection)
**Group:** _SessionXX-GroupY_ · **Last updated:** 2025-10-05

This repo contains:
- **Base dataset preprocessing** (school-provided emails)
- **Dataset 3 (Phishing URLs)** builder + quick baselines
- Reproducible **Conda** environment
- Ready-to-run **Windows commands**

---

## Quick start (Windows)
```bash
# 1) Create env
conda env create -f environment.yml
conda activate ai4cyber

# 2) Preprocess the base emails (put your raw CSV at data/base/emails.csv)
python scripts/preprocess_base_emails.py

# 3) Build Dataset 3 (downloads OpenPhish & Tranco, then featurizes)
python dataSet3_urls/scripts/build_dataset3_urls.py

# 4) Run quick baselines (classification + clustering) on Dataset 3
python dataSet3_urls/scripts/sample_eval_urls.py
```

Outputs:
- `data/processed_base/base_emails_processed.csv`
- `data/processed_base/base_emails_train.csv`
- `data/processed_base/base_emails_test.csv`
- `dataSet3_urls/processed/dataset3_urls.csv`

> For the assignment submission ZIP of datasets, include **only** the processed files actually used by your final models.

---

## Structure
```
CTI/
  .gitignore
  environment.yml
  README.md
  data/
    base/                # put the raw school dataset here as emails.csv
    processed_base/      # auto-created outputs
  dataSet3_urls/
    raw/
    processed/
    scripts/
      build_dataset3_urls.py
      features_url.py
      sample_eval_urls.py
    LICENSES_AND_SOURCES.md
  scripts/
    preprocess_base_emails.py
    tfidf_train_eval_base.py
  docs/
    report_template.md
    dataset3_urls_readme.md
```
