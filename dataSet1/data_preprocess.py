import pandas as pd
import numpy as np
import re
import string
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data (run once)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def basic_cleaning(self, text):
        """Basic text cleaning"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_special_chars(self, text):
        """Remove special characters and numbers"""
        # Keep only alphabets and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text
    
    def remove_urls_emails(self, text):
        """Remove URLs and email addresses"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        return text
    
    def remove_html_tags(self, text):
        """Remove HTML tags"""
        text = re.sub(r'<.*?>', '', text)
        return text
    
    def remove_stopwords(self, text):
        """Remove stop words"""
        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        return ' '.join(words)
    
    def apply_stemming(self, text):
        """Apply stemming to reduce words to root form"""
        words = text.split()
        stemmed = [self.stemmer.stem(word) for word in words]
        return ' '.join(stemmed)
    
    def apply_lemmatization(self, text):
        """Apply lemmatization (better than stemming)"""
        words = word_tokenize(text)
        lemmatized = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized)
    
    def preprocess_light(self, text):
        """Light preprocessing - fast and simple"""
        text = self.basic_cleaning(text)
        text = self.remove_special_chars(text)
        return text
    
    def preprocess_medium(self, text):
        """Medium preprocessing - balanced approach"""
        text = self.basic_cleaning(text)
        text = self.remove_urls_emails(text)
        text = self.remove_html_tags(text)
        text = self.remove_special_chars(text)
        text = self.remove_stopwords(text)
        return text
    
    def preprocess_heavy(self, text):
        """Heavy preprocessing - thorough cleaning"""
        text = self.basic_cleaning(text)
        text = self.remove_urls_emails(text)
        text = self.remove_html_tags(text)
        text = self.remove_special_chars(text)
        text = self.remove_stopwords(text)
        text = self.apply_stemming(text)  # or use apply_lemmatization(text)
        return text

# Example usage with your dataset
def preprocess_spam_data(csv_file, preprocessing_level='medium'):
    """
    Preprocess spam dataset
    preprocessing_level: 'light', 'medium', or 'heavy'
    """
    
    # Load data
    data = pd.read_csv(csv_file)
    
    # Detect columns
    if 'v1' in data.columns and 'v2' in data.columns:
        text_col, label_col = 'v2', 'v1'
    elif 'text' in data.columns and 'label' in data.columns:
        text_col, label_col = 'text', 'label'
    else:
        text_col, label_col = data.columns[1], data.columns[0]
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Choose preprocessing level
    if preprocessing_level == 'light':
        preprocess_func = preprocessor.preprocess_light
    elif preprocessing_level == 'heavy':
        preprocess_func = preprocessor.preprocess_heavy
    else:  # medium
        preprocess_func = preprocessor.preprocess_medium
    
    # Apply preprocessing
    print(f"Applying {preprocessing_level} preprocessing...")
    data['cleaned_text'] = data[text_col].apply(preprocess_func)
    
    # Remove empty texts
    data = data[data['cleaned_text'].str.strip() != '']
    
    # Convert labels
    data['label_numeric'] = data[label_col].map({'ham': 0, 'spam': 1})
    
    hotwords = {
    "free", "cash", "loan", "prize", "viagra", 
    "credit", "credit card", "winner", "guaranteed", 
    "sale", "luck", "offer", "money", "click", 
    "limited", "urgent", "win", "cheap", "deal", 
    "bonus", "gift"
    }
    
    def has_emoji(text):
        return any(not c.isascii() for c in str(text))

    def detect_hotwords(text):
        text = str(text).lower()
        matched = [hw for hw in hotwords if re.search(rf"\b{re.escape(hw)}\b", text)]
        return ", ".join(matched) if matched else "None"

    def has_hotword(text):
        words = set(text.lower().split())
        return int(any(hw in words for hw in hotwords))
    
    def extract_email(text):
        match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', str(text))
        return match.group(0) if match else "N/A"
    
    data['Label'] = data[label_col]
    data['Subject'] = data[text_col].apply(lambda x: " ".join(str(x).split()[:3]) if isinstance(x, str) else "N/A")  # first 3 words
    data['Content'] = data['cleaned_text']
    data['Sender Email'] = data[text_col].apply(extract_email)
    data['Use emoji'] = data[text_col].apply(lambda x: int(has_emoji(x)))
    data['Use exclamation mark?'] = data[text_col].apply(lambda x: int("!" in str(x)))
    data['Hotword'] = data['cleaned_text'].apply(has_hotword)
    
    print(f"Original dataset: {len(data)} samples")
    print(f"After cleaning: {len(data)} samples")
    print(f"Label distribution:\n{data['label_numeric'].value_counts()}")
    
    return data, text_col, label_col

# Advanced preprocessing options
def advanced_preprocessing():
    """Additional preprocessing techniques"""
    
    # 1. Handle class imbalance
    from sklearn.utils import resample
    
    def balance_dataset(data):
        # Separate classes
        ham_data = data[data['label_numeric'] == 0]
        spam_data = data[data['label_numeric'] == 1]
        
        # Upsample minority class
        if len(ham_data) > len(spam_data):
            spam_upsampled = resample(spam_data, 
                                    replace=True, 
                                    n_samples=len(ham_data), 
                                    random_state=42)
            balanced_data = pd.concat([ham_data, spam_upsampled])
        else:
            ham_upsampled = resample(ham_data, 
                                   replace=True, 
                                   n_samples=len(spam_data), 
                                   random_state=42)
            balanced_data = pd.concat([spam_data, ham_upsampled])
        
        return balanced_data
    
    # 2. Feature extraction options
    def get_vectorizer_options():
        vectorizers = {
            'tfidf': TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2)),
            'count': CountVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2)),
            'tfidf_char': TfidfVectorizer(max_features=5000, analyzer='char', ngram_range=(2,4))
        }
        return vectorizers
    
    # 3. Text statistics features
    def extract_text_features(text):
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'dollar_count': text.count('$'),
        }
        return features

# Complete preprocessing pipeline
def complete_preprocessing_pipeline(csv_file):
    """Complete preprocessing pipeline"""
    
    # Load and preprocess
    data = preprocess_spam_data(csv_file, 'medium')
    
    # Extract additional features
    preprocessor = TextPreprocessor()
    
    # Add text statistics
    data['text_length'] = data['cleaned_text'].str.len()
    data['word_count'] = data['cleaned_text'].str.split().str.len()
    data['uppercase_ratio'] = data['cleaned_text'].apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
    )
    
    # Final dataset
    X_text = data['cleaned_text'].values
    X_features = data[['text_length', 'word_count', 'uppercase_ratio']].values
    y = data['label_numeric'].values
    
    return X_text, X_features, y, data

def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', str(text))
    return match.group(0) if match else "N/A"

# Simple usage example
if __name__ == "__main__":
    data = preprocess_spam_data('spam_email.csv', 'medium')

    output_file = "preprocessed_spam.csv"
    data[['Label', 'Sender Email', 'Subject', 'Content',
          'Use emoji', 'Use exclamation mark?', 'Hotword',
          'Text Length', 'Word Count', 'Uppercase Ratio',
          'Digit Count', 'URL Count', 'Special Char Count']].to_csv(output_file, index=False)

    print(f"\nProcessed data saved to {output_file}")
    print(data.head(5))
