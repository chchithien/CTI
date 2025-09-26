import pandas as pd
import re
import matplotlib.pyplot as plt


def clean_message(text: str) -> str:
    if pd.isna(text):
        return ""

    
 # (from, to, subject, cc, forwarded by, sent by, etc.)
    text = re.sub(r'^(from|to|subject|cc|bcc|sent by).*?:.*', '', text, flags=re.IGNORECASE | re.MULTILINE)

    # remove table borders (| --- | etc.)
    text = re.sub(r'\|.*\|', '', text)

    # remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()

    # remove consecutive 3 or more symbols (., - , _ , space, etc.)
    text = re.sub(r'[.\-_ ]{3,}', ' ', text)

    # remove table borders (| --- | etc.)
    text = re.sub(r'\|.*\|', '', text)

    # delete time formats like 10:30 am, 10:30, 10:30pm, 10：30
    text = re.sub(r'\d{1,2} ?[:：] ?\d{2} ?(am|pm)?', '', text, flags=re.IGNORECASE)

    # delete dates
    text = re.sub(r'\d{1,2} ?/ ?\d{1,2} ?/ ?\d{2,4}', '', text)

    text = re.sub(r'/\s*\w+\s*/\s*\w+', '', text, flags=re.IGNORECASE)


    text = text.lower()


    return text

def clean_dataset(input_file: str, output_file: str):
    df = pd.read_csv(input_file)

    df = df.dropna(subset=["Message", "Spam/Ham"])
    df["Message"] = df["Message"].apply(clean_message)
    df = df[df["Message"].str.strip() != ""]
    df = df.drop_duplicates()

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
    print(f"clean complete, {len(df)} rows retained, saved to: {output_file}")

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
