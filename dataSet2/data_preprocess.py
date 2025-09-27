import pandas as pd
import re
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("omw-1.4")

def remove_headers(text: str) -> str:
    text = re.sub(r'forwarded by .*? on', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^(from|to|subject|cc|bcc|sent by|forwarded by|fyi).*?:.*',
                  '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'\bfyi\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^(hou|etc).*?\<.*\>', '', text, flags=re.IGNORECASE | re.MULTILINE)
    return text

def remove_tables(text: str) -> str:
    text = re.sub(r'\|.*\|', '', text)
    text = re.sub(r'[.\-_ ]{3,}', ' ', text)
    return text

def remove_emails(text: str) -> str:
    return re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

def remove_dates_times(text: str) -> str:
    text = re.sub(r'\d{1,2} ?[:：] ?\d{2} ?(am|pm)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bam\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bpm\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d{1,2} ?[/-] ?\d{1,2} ?[/-] ?\d{2,4}', '', text)
    text = re.sub(r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', '', text)
    text = re.sub(r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[ .,-]*\d{2,4}\b',
                  '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{2,4}[ .,-]*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b',
                  '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(19|20)\d{2}\b', '', text)
    return text

def remove_symbols(text: str) -> str:
    return re.sub(r"[^\w\s!?$%']", " ", text)

def normalize_spaces(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def remove_numbers(text: str) -> str:
    # Normalize money format: allow "2 $" -> "$2"
    text = re.sub(r'(\d+)\s*\$', r'$\1', text)   # Convert "2 $" -> "$2"
    text = re.sub(r'\$\s*(\d+)', r'$\1', text)   # Convert "$ 2" -> "$2"

    # Keep percentage
    text = re.sub(r'(\d+)\s*%', r'\1%', text)

    # Remove year (1900–2099)
    text = re.sub(r'\b(19|20)\d{2}\b', '', text)

    # Remove phone numbers (123-456-7890 / 123 456 7890)
    text = re.sub(r'\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b', '', text)

    # Remove long numbers (>=5 digits, e.g. meter ID, contract number)
    text = re.sub(r'\b\d{5,}\b', '', text)

    # Remove standalone small numbers (1~4 digits, not money/percentage)
    text = re.sub(r'(?<!\$)\b\d{1,4}(?!%|\d)\b', '', text)

    return text

# initial stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def apply_stemming(text: str) -> str:
    words = text.split()
    stemmed = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed)

def apply_lemmatization(text: str) -> str:
    words = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized)

def clean_message(text: str) -> str:
    if pd.isna(text):
        return ""

    text = remove_headers(text)
    text = remove_tables(text)
    text = remove_emails(text)
    text = remove_dates_times(text)
    text = remove_symbols(text)
    text = normalize_spaces(text)

    text = apply_lemmatization(text)
    # text = apply_stemming(text) for heavy preprocessing

    return text

def clean_dataset(input_file: str, output_file: str):
    df = pd.read_csv(input_file)
    df = df.drop(columns=["Date"], errors="ignore")
    df = df.dropna(subset=["Message", "Spam/Ham"])
    df["Message"] = df["Message"].apply(remove_numbers)
    df["Message"] = df["Message"].apply(clean_message)

    if "Subject" in df.columns:
        df["Subject"] = df["Subject"].apply(clean_message)

    df = df[df["Message"].str.strip() != ""]
    df = df.drop_duplicates()
    df["Spam/Ham"] = df["Spam/Ham"].str.lower().map({'ham': 0, 'spam': 1})

    df["length"] = df["Message"].apply(len)
    total_before = len(df)
    df = df[df["length"] >= 5]
    removed_too_short = total_before - len(df)

    Q1 = df["length"].quantile(0.25)
    Q3 = df["length"].quantile(0.75)
    IQR = Q3 - Q1
    df_before_iqr = len(df)
    df = df[~((df["length"] < (Q1 - 1.5 * IQR)) | (df["length"] > (Q3 + 1.5 * IQR)))]
    removed_iqr = df_before_iqr - len(df)

    df = df.drop(columns=["length"])

    print(f"Total rows before filtering: {total_before}")
    print(f"Rows removed because length < 5: {removed_too_short}")
    print(f"Rows removed by IQR filtering: {removed_iqr}")
    print(f"Total rows after filtering: {len(df)}")

    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Clean complete, {len(df)} rows retained, saved to: {output_file}")

def analyze_cleaned_data(file: str):
    df = pd.read_csv(file)
    df = df[df["Message"].notna()]
    df["length"] = df["Message"].apply(len)
    print("Length Describe:")
    print(df["length"].describe())

    plt.figure(figsize=(10, 4))
    plt.boxplot(df["length"], vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.title("Boxplot of Cleaned Email Message Lengths")
    plt.xlabel("Message Length")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    input_file = "Desktop\\CTI\\dataSet2\\enron_spam_data.csv"
    output_file = "emails_clean.csv"
    clean_dataset(input_file, output_file)
    analyze_cleaned_data(output_file)
