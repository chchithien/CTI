
# Dataset 3 — Phishing URLs (OpenPhish + Tranco)

**Collection date (AEST):** <fill with run date>

**Task:** Binary classification: phishing (1) vs benign (0).

**Sources:**
- Malicious: OpenPhish community feed (`feed.txt`)
- Benign: Tranco Top Sites (daily list)

**Labels:** `label` ∈ {0=benign, 1=phish}

**Columns:**
- `url`, `label`
- Engineered lexical features: `len_url`, `len_host`, `len_path`, `count_hyphen`, `count_at`, `count_question`, `count_equals`, `count_digits`, `count_dots`, `has_ip_host`, `host_num_labels`, `tld_len`, `is_suspicious_tld`, `contains_login`, `contains_verify`, `contains_update`, `contains_secure`, `contains_bank`.

## Build
```bash
conda activate ai4cyber
python dataSet3_urls/scripts/build_dataset3_urls.py
```
Outputs CSV at: `dataSet3_urls/processed/dataset3_urls.csv`
