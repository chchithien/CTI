
import os, random, datetime, requests, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
PROC = ROOT / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

TODAY = datetime.date.today().isoformat()

OPENPHISH_URL = "https://openphish.com/feed.txt"   # malicious
TRANCO_LIST_URL = "https://tranco-list.eu/top-1m.csv"  # benign

def download(url, path):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    path.write_bytes(r.content)

def load_openphish(path: Path):
    urls = [line.strip() for line in path.read_text("utf-8", errors="ignore").splitlines() if line.strip()]
    return pd.DataFrame({"url": urls, "label": 1})

def load_tranco(path: Path, n=10000):
    df = pd.read_csv(path, header=None, names=["rank","domain"])
    df = df.sample(min(n, len(df)), random_state=42)
    df["url"] = "http://" + df["domain"].astype(str) + "/"
    df["label"] = 0
    return df[["url","label"]]

if __name__ == "__main__":
    op_path = RAW / f"openphish_{TODAY}.txt"
    tr_path = RAW / f"tranco_{TODAY}.csv"

    if not op_path.exists():
        print("Downloading OpenPhish...")
        download(OPENPHISH_URL, op_path)

    if not tr_path.exists():
        print("Downloading Tranco...")
        download(TRANCO_LIST_URL, tr_path)

    df_bad = load_openphish(op_path)
    df_good = load_tranco(tr_path, n=len(df_bad))

    df = pd.concat([df_bad, df_good], ignore_index=True).drop_duplicates("url")
    from features_url import featurize_urls
    X = featurize_urls(df["url"])
    out = pd.concat([df, X], axis=1)
    out_path = PROC / "dataset3_urls.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
