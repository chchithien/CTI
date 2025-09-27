import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    def __init__(self, data):
        self.data = data.copy()
        self.original_shape = data.shape
        self.cleaning_log = []
    
    def remove_duplicates(self, subset_columns=None, keep='first'):
        # Remove duplicate rows from the dataset
        initial_count = len(self.data)
        
        # Check for exact duplicates
        exact_duplicates = self.data.duplicated().sum()
        print(f"Exact duplicates found: {exact_duplicates}")
        
        # Check for duplicates in specific columns if provided
        if subset_columns:
            subset_duplicates = self.data.duplicated(subset=subset_columns).sum()
            print(f"Duplicates in {subset_columns}: {subset_duplicates}")
            
            # Remove duplicates based on subset
            self.data = self.data.drop_duplicates(subset=subset_columns, keep=keep)
        else:
            # Remove exact duplicates
            self.data = self.data.drop_duplicates(keep=keep)
        
        final_count = len(self.data)
        removed_count = initial_count - final_count
        
        print(f"Removed {removed_count} duplicate rows")
        print(f"Dataset size: {initial_count} → {final_count}")
        
        self.cleaning_log.append(f"Removed {removed_count} duplicates")
        return self.data
    
    def detect_outliers_iqr(self, column):
        # Detect outliers using IQR method
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = self.data[(self.data[column] < lower_bound) | 
                            (self.data[column] > upper_bound)]
        
        return outliers, lower_bound, upper_bound
    
    def detect_outliers_zscore(self, column, threshold=3):
        # Detect outliers using Z-score method
        z_scores = np.abs(stats.zscore(self.data[column].dropna()))
        outlier_indices = np.where(z_scores > threshold)[0]
        outliers = self.data.iloc[outlier_indices]
        
        return outliers, threshold
    
    def plot_boxplots_with_outliers(self, numeric_columns, target_column=None, figsize=(15, 10)):
        # Create boxplots for numeric columns and highlight outliers
        n_cols = min(3, len(numeric_columns))
        n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        outlier_summary = {}
        
        for i, column in enumerate(numeric_columns):
            if i < len(axes):
                # Create boxplot
                if target_column and target_column in self.data.columns:
                    sns.boxplot(data=self.data, x=target_column, y=column, ax=axes[i])
                    axes[i].set_title(f'{column} by {target_column}')
                else:
                    sns.boxplot(data=self.data, y=column, ax=axes[i])
                    axes[i].set_title(f'{column} Distribution')
                
                # Detect outliers using IQR
                outliers, lower_bound, upper_bound = self.detect_outliers_iqr(column)
                outlier_count = len(outliers)
                
                # Add outlier information to plot
                axes[i].text(0.02, 0.98, f'Outliers: {outlier_count}', 
                            transform=axes[i].transAxes, 
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                outlier_summary[column] = {
                    'count': outlier_count,
                    'percentage': (outlier_count / len(self.data)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
        
        # Hide extra subplots
        for i in range(len(numeric_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Print outlier summary
        for column, info in outlier_summary.items():
            print(f"{column}:")
            print(f"  • Outliers: {info['count']} ({info['percentage']:.2f}%)")
            print(f"  • Bounds: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")
            print()
        
        return outlier_summary
    
    def handle_outliers(self, column, method='remove', custom_bounds=None):
        # Handle outliers in a specific column
        initial_count = len(self.data)
        
        if custom_bounds:
            lower_bound, upper_bound = custom_bounds
        else:
            _, lower_bound, upper_bound = self.detect_outliers_iqr(column)
        
        if method == 'remove':
            # Remove outliers
            self.data = self.data[
                (self.data[column] >= lower_bound) & 
                (self.data[column] <= upper_bound)
            ]
            removed_count = initial_count - len(self.data)
            print(f"Removed {removed_count} outliers")
            self.cleaning_log.append(f"Removed {removed_count} outliers from {column}")
            
        elif method == 'cap':
            # Cap outliers to bounds
            self.data[column] = np.clip(self.data[column], lower_bound, upper_bound)
            print(f"Capped outliers to bounds [{lower_bound:.2f}, {upper_bound:.2f}]")
            self.cleaning_log.append(f"Capped outliers in {column}")
            
        elif method == 'transform':
            # Log transformation for positive values
            if self.data[column].min() > 0:
                self.data[f'{column}_log'] = np.log1p(self.data[column])
                print(f"Applied log transformation (new column: {column}_log)")
                self.cleaning_log.append(f"Log transformed {column}")
            else:
                print("Cannot apply log transformation (negative values present)")
        
        return self.data
    
    def auto_outlier_handling(self, numeric_columns, outlier_threshold=5.0):
        # Automatic outlier handling based on percentage threshold
        for column in numeric_columns:
            outliers, lower_bound, upper_bound = self.detect_outliers_iqr(column)
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(self.data)) * 100
            
            if outlier_count > 0:
                print(f"\nColumn: {column}")
                print(f"   Outliers: {outlier_count} ({outlier_percentage:.2f}%)")
                print(f"   Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
                print("   Sample outlier values:", outliers[column].head().tolist())
                
                # Automatic decision logic
                if outlier_percentage > 20:
                    # Too many outliers - cap them instead of removing
                    print("   Capping outliers (>20% of data)")
                    self.handle_outliers(column, method='cap')
                elif outlier_percentage > outlier_threshold:
                    # Moderate outliers - remove them
                    print(f"   Removing outliers (>{outlier_threshold}% of data)")
                    self.handle_outliers(column, method='remove')
                elif column in ['Text Length', 'Word Count'] and outlier_percentage > 2:
                    # For text-related features, be more aggressive
                    print("   Removing text-related outliers (>2% of data)")
                    self.handle_outliers(column, method='remove')
                else:
                    # Few outliers - keep them
                    print(f"   Keeping outliers (<{outlier_threshold}% of data)")
            else:
                print(f"\nNo outliers found in {column}")
    
    def clean_data_pipeline(self, numeric_columns, target_column=None, 
                            duplicate_subset=None, outlier_threshold=5.0):
        # Complete automatic data cleaning pipeline
        # Remove duplicates
        self.remove_duplicates(subset_columns=duplicate_subset)
        
        # Visualize outliers
        outlier_summary = self.plot_boxplots_with_outliers(numeric_columns, target_column)
        
        # Handle outliers automatically
        self.auto_outlier_handling(numeric_columns, outlier_threshold)
        
        # Final summary
        print("Cleaning Summary:")
        print(f"Original shape: {self.original_shape}")
        print(f"Final shape: {self.data.shape}")
        print(f"Rows removed: {self.original_shape[0] - self.data.shape[0]}")
        print(f"Data retention: {(self.data.shape[0]/self.original_shape[0])*100:.2f}%")
        print("\nCleaning steps performed:")
        for i, step in enumerate(self.cleaning_log, 1):
            print(f"  {i}. {step}")
        
        return self.data
    
    def get_cleaned_data(self):
        # Return the cleaned dataset
        return self.data
    
    def save_cleaned_data(self, filename):
        # Save cleaned data to CSV
        self.data.to_csv(filename, index=False)
        print(f"\nCleaned data saved to: {filename}")

# Usage example with your preprocessed data
def clean_preprocessed_spam_data(csv_file, outlier_threshold=5.0):
    # Clean your preprocessed spam data automatically
    
    # Load preprocessed data
    data = pd.read_csv(csv_file)
    print(f"Loaded data shape: {data.shape}")
    
    # Initialize cleaner
    cleaner = DataCleaner(data)
    
    # Define numeric columns for outlier detection
    numeric_columns = ['Text Length', 'Word Count', 'Uppercase Ratio', 
                        'Digit Count', 'URL Count', 'Special Char Count']
    
    # Filter to only existing numeric columns
    existing_numeric_cols = [col for col in numeric_columns if col in data.columns]
    print(f"Numeric columns found: {existing_numeric_cols}")
    
    # Clean data automatically
    cleaned_data = cleaner.clean_data_pipeline(
        numeric_columns=existing_numeric_cols,
        target_column='Label',  # for grouped boxplots
        duplicate_subset=['Content'],  # check duplicates based on content
        outlier_threshold=outlier_threshold  # percentage threshold for outlier removal
    )
    
    return cleaned_data, cleaner

# Non-interactive cleaning function
def auto_clean_data(csv_file, output_file='cleaned_spam_data.csv', outlier_threshold=5.0):
    # Automatic data cleaning without user interaction
    
    data = pd.read_csv(csv_file)
    cleaner = DataCleaner(data)
    
    # Define columns
    numeric_columns = ['Text Length', 'Word Count', 'Uppercase Ratio', 
                        'Digit Count', 'URL Count', 'Special Char Count']
    existing_numeric_cols = [col for col in numeric_columns if col in data.columns]
    
    # Auto clean
    cleaned_data = cleaner.clean_data_pipeline(
        numeric_columns=existing_numeric_cols,
        target_column='Label',
        duplicate_subset=['Content'],
        outlier_threshold=outlier_threshold
    )
    
    # Save results
    cleaner.save_cleaned_data(output_file)
    
    return cleaned_data

# Complete example with CSV output
def complete_cleaning_with_output(input_file, output_file="cleaned_spam_data.csv", outlier_threshold=5.0):
    # Complete automatic cleaning pipeline with CSV output
    try:
        # Load and clean data automatically
        cleaned_data, cleaner = clean_preprocessed_spam_data(input_file, outlier_threshold)
        
        # Save to CSV
        cleaner.save_cleaned_data(output_file)
        
        # Display summary statistics
        print(f"Final dataset shape: {cleaned_data.shape}")
        print(f"Columns: {list(cleaned_data.columns)}")
        
        # Show label distribution
        if 'Label' in cleaned_data.columns:
            print(f"\nLabel distribution:")
            print(cleaned_data['Label'].value_counts())
        
        # Show basic statistics for numeric columns
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\nNumeric columns statistics:")
            print(cleaned_data[numeric_cols].describe())
        
        # Show first few rows
        print(f"\nFirst 5 rows of cleaned data:")
        print(cleaned_data.head())
        
        print(f"\nSUCCESS: Cleaned data saved to '{output_file}'")
        return cleaned_data
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return None

if __name__ == "__main__":
    # Replace with your preprocessed file path
    input_file = "preprocessed_spam.csv"
    output_file = "final_cleaned_spam_data.csv"
    
    # Automatic cleaning with default settings
    cleaned_data = complete_cleaning_with_output(input_file, output_file, outlier_threshold=5.0)