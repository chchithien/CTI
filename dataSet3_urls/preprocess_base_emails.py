

import re, string
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

IN_PATH = Path("data/base/emails.csv")        # place raw school dataset here
OUT_DIR = Path("data/processed_base")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def guess_columns(df):
    text_candidates = ["Message", "message", "Text", "text", "EmailText", "Email_Body", "Body", "content"]
    label_candidates = ["Spam/Ham", "SpamHam", "spam_ham", "label", "Label", "target", "Category", "category", "is_spam"]
    subj_candidates = ["Subject", "subject", "Title", "title"]
    text_col = label_col = subj_col = None
    cols = set(df.columns)
    for c in text_candidates:
        if c in cols: text_col = c; break
    for c in label_candidates:
        if c in cols: label_col = c; break
    for c in subj_candidates:
        if c in cols: subj_col = c; break
    if text_col is None and subj_col is not None:
        for c in df.columns:
            if c != subj_col and df[c].dtype == object and c.lower() not in ["label", "spam/ham", "category", "target"]:
                text_col = c; break
    return subj_col, text_col, label_col

URL_RE = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
EMAIL_RE = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
NUM_RE = re.compile(r'\b\d+\b')

def clean_text(s):
    if not isinstance(s, str): return ""
    s = s.lower()
    s = URL_RE.sub(" ", s)
    s = EMAIL_RE.sub(" ", s)
    s = NUM_RE.sub(" ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_label(v):
    if pd.isna(v): return None
    s = str(v).strip().lower()
    if s.isdigit():
        i = int(s); 
        return i if i in (0,1) else (1 if i>0 else 0)
    if s in ["spam","yes","true","1","bad"]: return 1
    if s in ["ham","no","false","0","good"]: return 0
    return 1 if "spam" in s else 0

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Raw file not found: {IN_PATH}. Put the school CSV there as emails.csv")
    df_raw = pd.read_csv(IN_PATH)
    subj_col, text_col, label_col = guess_columns(df_raw)

    if text_col is None:
        for c in df_raw.columns:
            if str(c).lower() in ["message","body","text","email_body","content"]:
                text_col = c; break

    def combine_text(row):
        parts = []
        if subj_col and isinstance(row.get(subj_col, ""), str): parts.append(str(row[subj_col]))
        if text_col and isinstance(row.get(text_col, ""), str): parts.append(str(row[text_col]))
        if not parts:
            objs = [str(row[c]) for c in df_raw.columns if df_raw[c].dtype == object]
            parts = objs
        return " ".join([p for p in parts if isinstance(p, str)])

    combined_text = df_raw.apply(combine_text, axis=1)
    labels = df_raw[label_col].apply(normalize_label) if label_col is not None else pd.Series([0]*len(df_raw))

    df = pd.DataFrame({
        "subject": df_raw[subj_col] if subj_col else "",
        "message": df_raw[text_col] if text_col else "",
        "text": combined_text,
        "label": labels
    })
    df["text"] = df["text"].astype(str)
    df["text_clean"] = df["text"].apply(clean_text)
    df = df[df["text_clean"].str.len() > 0].copy()
    df = df.drop_duplicates(subset=["text_clean"])
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    df["char_len"] = df["text_clean"].str.len()
    df["word_count"] = df["text_clean"].str.split().apply(len)

    train_df, test_df = train_test_split(
        df[["text_clean","label"]],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "base_emails_processed.csv", index=False)
    train_df.to_csv(OUT_DIR / "base_emails_train.csv", index=False)
    test_df.to_csv(OUT_DIR / "base_emails_test.csv", index=False)

    # small summary
    (OUT_DIR / "base_emails_summary.txt").write_text(
        f"rows_after_clean={len(df)}\\ntrain={len(train_df)} test={len(test_df)}\\nclass_counts={df['label'].value_counts().to_dict()}"
    )
    print("Done. Saved to data/processed_base/")

if __name__ == "__main__":
    main()
