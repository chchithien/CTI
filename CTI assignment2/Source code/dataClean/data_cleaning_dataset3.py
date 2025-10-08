import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

class SpamDataProcessor:
    def __init__(self, csv_file):
        """Initialize with CSV file path"""
        self.data = pd.read_csv(csv_file)
        print(f"Loaded dataset: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
    
    def clean_text(self, text):
        """Basic text cleaning"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        
        return text
    
    def extract_features(self):
        """Extract useful features from the messages"""
        
        # Clean the Message column
        self.data['Cleaned_Message'] = self.data['Message'].apply(self.clean_text)
        
        # Extract features
        self.data['Text_Length'] = self.data['Cleaned_Message'].str.len()
        self.data['Word_Count'] = self.data['Cleaned_Message'].apply(lambda x: len(str(x).split()))
        self.data['Uppercase_Ratio'] = self.data['Message'].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
        )
        self.data['Digit_Count'] = self.data['Message'].apply(lambda x: sum(c.isdigit() for c in str(x)))
        self.data['Exclamation_Count'] = self.data['Message'].apply(lambda x: str(x).count('!'))
        self.data['Question_Count'] = self.data['Message'].apply(lambda x: str(x).count('?'))
        self.data['Dollar_Count'] = self.data['Message'].apply(lambda x: str(x).count('$'))
        
        # URL detection
        self.data['URL_Count'] = self.data['Message'].apply(
            lambda x: len(re.findall(r'http[s]?://|www\.', str(x)))
        )
        
        # Email detection
        self.data['Email_Count'] = self.data['Message'].apply(
            lambda x: len(re.findall(r'[\w\.-]+@[\w\.-]+', str(x)))
        )
        
        print(f"\nFeatures extracted successfully!")
        return self.data
    
    def remove_duplicates(self):
        """Remove duplicate messages"""
        initial_count = len(self.data)
        
        # Remove exact duplicates based on message content
        self.data = self.data.drop_duplicates(subset=['Message'], keep='first')
        
        removed = initial_count - len(self.data)
        print(f"\nRemoved {removed} duplicate messages")
        print(f"Dataset size: {initial_count} → {len(self.data)}")
        
        return self.data
    
    def handle_outliers(self, column, method='remove'):
        """Handle outliers using IQR method"""
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers = self.data[(self.data[column] < lower_bound) | (self.data[column] > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(self.data)) * 100
        
        print(f"\n{column}:")
        print(f"  Outliers: {outlier_count} ({outlier_pct:.2f}%)")
        print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        if method == 'remove' and outlier_pct < 10:
            # Only remove if less than 10% are outliers
            self.data = self.data[(self.data[column] >= lower_bound) & (self.data[column] <= upper_bound)]
            print(f"  Action: Removed {outlier_count} outliers")
        elif method == 'cap':
            # Cap outliers to bounds
            self.data[column] = np.clip(self.data[column], lower_bound, upper_bound)
            print(f"  Action: Capped outliers")
        else:
            print(f"  Action: Kept outliers (too many or method='keep')")
        
        return self.data
    
    def clean_outliers(self):
        """Automatically clean outliers from all numeric features"""
        numeric_features = ['Text_Length', 'Word_Count', 'Uppercase_Ratio', 
                          'Digit_Count', 'Exclamation_Count', 'URL_Count']
        
        print("\n=== Handling Outliers ===")
        for feature in numeric_features:
            if feature in self.data.columns:
                # Use cap method for ratio features, remove for others
                method = 'cap' if 'Ratio' in feature else 'remove'
                self.handle_outliers(feature, method=method)
        
        return self.data
    
    def process(self, remove_duplicates=True, handle_outliers=True):
        """Complete processing pipeline"""
        print("\n=== Starting Data Processing ===")
        
        # Extract features
        self.extract_features()
        
        # Remove duplicates
        if remove_duplicates:
            self.remove_duplicates()
        
        # Handle outliers
        if handle_outliers:
            self.clean_outliers()
        
        # Remove rows with empty messages
        initial_count = len(self.data)
        self.data = self.data[self.data['Cleaned_Message'].str.strip() != '']
        removed = initial_count - len(self.data)
        if removed > 0:
            print(f"\nRemoved {removed} empty messages")
        
        print(f"\n=== Processing Complete ===")
        print(f"Final dataset shape: {self.data.shape}")
        print(f"\nLabel distribution:")
        print(self.data['Spam/Ham'].value_counts())
        
        return self.data
    
    def save(self, output_file='cleaned_spam_data.csv'):
        """Save processed data to CSV"""
        # Select important columns
        output_cols = ['Message ID', 'Subject', 'Cleaned_Message', 'Spam/Ham',
                      'Text_Length', 'Word_Count', 'Uppercase_Ratio', 
                      'Digit_Count', 'Exclamation_Count', 'Question_Count',
                      'Dollar_Count', 'URL_Count', 'Email_Count']
        
        # Only include columns that exist
        output_cols = [col for col in output_cols if col in self.data.columns]
        
        self.data[output_cols].to_csv(output_file, index=False)
        print(f"\nProcessed data saved to: {output_file}")
        
        # Show sample
        print("\nFirst 3 rows:")
        print(self.data[output_cols].head(3))
        
        return output_file
    
    def get_summary(self):
        """Display summary statistics"""
        print("\n=== Dataset Summary ===")
        print(f"Total messages: {len(self.data)}")
        print(f"\nSpam/Ham distribution:")
        print(self.data['Spam/Ham'].value_counts())
        
        numeric_cols = ['Text_Length', 'Word_Count', 'Uppercase_Ratio', 
                       'Digit_Count', 'Exclamation_Count']
        numeric_cols = [col for col in numeric_cols if col in self.data.columns]
        
        if numeric_cols:
            print(f"\nNumeric features statistics:")
            print(self.data[numeric_cols].describe())


# Simple usage function
def process_spam_data(input_file, output_file='cleaned_spam_data.csv'):
    """
    Simple one-function call to process spam data
    
    Parameters:
    - input_file: path to your CSV file
    - output_file: path to save cleaned data
    """
    # Initialize processor
    processor = SpamDataProcessor(input_file)
    
    # Process data
    cleaned_data = processor.process(
        remove_duplicates=True,
        handle_outliers=True
    )
    
    # Save results
    processor.save(output_file)
    
    # Show summary
    processor.get_summary()
    
    return cleaned_data


# Example usage
if __name__ == "__main__":
    # Replace with your file path
    input_file = "formatted_school_dataset.csv"
    output_file = "cleaned_school_dataset.csv"
    
    # Process the data
    cleaned_data = process_spam_data(input_file, output_file)
    
    print("\n✓ Done! Your cleaned data is ready.")