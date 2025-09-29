import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# 读取数据
df = pd.read_csv("Desktop\\CTI\\dataSet2\\emails_clean.csv")

# 填充缺失值
df["Subject"] = df["Subject"].fillna("")
df["Message"] = df["Message"].fillna("")

# 定义 spam 关键词
spam_words = [
    "free", "money", "offer", "win", "click", "buy",
    "credit", "investment", "limited", "guarantee", "prize",
    "porn", "sex", "loan", "cash", "winner", "viagra",
    "cheap", "sale", "billion", "million", "access",
    "save", "bonus", "reward", "gift"
]


spam_groups = {
    "money_related": ["free", "money", "cash", "prize"],
    "click_related": ["click", "offer", "buy", "limited", "access"],

}



def extract_features_with_tfidf(df, max_features=3000):
    # -------- 1. 结构化 & 关键词特征 --------
    features = pd.DataFrame()

    
    features['subject_length'] = df['Subject'].apply(len)
    features['message_length'] = df['Message'].apply(len)
    features['num_words_subject'] = df['Subject'].apply(lambda x: len(x.split()))
    features['num_words_message'] = df['Message'].apply(lambda x: len(x.split()))
    features['is_subject_missing'] = (df['Subject'] == "").astype(int)
    features['num_exclamations'] = df['Message'].apply(lambda x: x.count('!'))
    features['num_questionmarks'] = df['Message'].apply(lambda x: x.count('?'))

    features['exclamation_ratio'] = features['num_exclamations'] / (features['message_length'] + 1)

    
    features['num_dollarsign'] = df['Message'].apply(lambda x: x.count('$'))

    features['num_number_dollar'] = df['Message'].apply(
        lambda x: len(re.findall(r'(\d+\s*\$)|(\$\s*\d+)', x))
    )




    features['num_percent'] = df['Message'].apply(lambda x: len(re.findall(r'[%]', x)))
    features['percent_ratio'] = features['num_percent'] / (features['message_length'] + 1)

    features['num_digits'] = df['Message'].apply(lambda x: sum(c.isdigit() for c in x))
    features['digit_ratio'] = features['num_digits'] / (features['message_length'] + 1)
    features['num_urls'] = df['Message'].apply(lambda x: len(re.findall(r'http|www', x.lower())))

    features['word_diversity'] = df['Message'].apply(lambda x: len(set(x.split())) / (len(x.split())+1))

    features['num_repeated_words'] = df['Message'].apply(
    lambda x: sum([1 for w in x.lower().split() if x.lower().split().count(w) > 2])
)
    


    # Spam 关键词统计
    combined_text = df["Subject"] + " " + df["Message"]
    for word in spam_words:
        features[f'word_{word}'] = combined_text.apply(
            lambda x: len(re.findall(rf'\b{word}\b', x, flags=re.IGNORECASE))
        )

    for group_name, words in spam_groups.items():
        features[f"group_{group_name}"] = combined_text.apply(
            lambda x: sum(len(re.findall(rf'\b{w}\b', x, flags=re.IGNORECASE)) for w in words)
        )
    # -------- 2. TF-IDF 特征 --------
    tfidf = TfidfVectorizer(stop_words="english", max_features=max_features)
    X_tfidf = tfidf.fit_transform(combined_text)

    # -------- 3. 合并 --------
    X = hstack([X_tfidf, features.values])
    y = df["Spam/Ham"].values

    # -------- 4. 返回结构化特征 DataFrame --------
    struct_df = features.copy()
    struct_df["Spam/Ham"] = y

    return X, y, tfidf, features.columns, struct_df

# ==== 特征可视化 ====
def visualize_struct_features(data, features, label_col="Spam/Ham"):
    # 各特征箱线图
    for col in features:
        plt.figure(figsize=(7,5))
        sns.boxplot(data=data, x=label_col, y=col)
        plt.title(f"{col} (0=Ham, 1=Spam)")
        plt.show()

    # 相关性热力图
    plt.figure(figsize=(14,10))
    corr = data[features.tolist() + [label_col]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

# ==== Top N 特征条形图 ====
def plot_top_correlated_features(data, label_col="Spam/Ham", top_n=15):
    corr = data.corr()[label_col].drop(label_col)
    top_features = corr.abs().sort_values(ascending=False).head(top_n)

    print(f"Top {top_n} correlated features with {label_col}:")
    print(corr[top_features.index].sort_values(key=abs, ascending=False))

    plt.figure(figsize=(10,6))
    sns.barplot(x=top_features.values, y=top_features.index, palette="coolwarm")
    plt.title(f"Top {top_n} Features Correlated with {label_col}")
    plt.xlabel("Correlation")
    plt.ylabel("Feature")
    plt.show()

# ==== 调用示例 ====
X, y, tfidf, struct_feature_names, struct_df = extract_features_with_tfidf(df)
print("特征矩阵大小:", X.shape)
print("标签大小:", y.shape)
print("结构特征列:", list(struct_feature_names))

# 可视化前15个最相关特征
plot_top_correlated_features(struct_df, "Spam/Ham", top_n=50)
