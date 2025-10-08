# Data Cleaning and Preprocessing Pipeline

This repository contains Python scripts for cleaning and preprocessing raw datasets.  
The workflow involves two main scripts:

1. **`data_cleaning.py`** — cleans the raw dataset (handles missing values, duplicates, etc.)
2. **`data_preprocess.py`** — preprocesses the cleaned data (feature encoding, normalization, splitting, etc.)

---

## 🧩 Project Structure

```
.
├── dataset_1.csv           # Raw input dataset
├── data_cleaning.py        # Data cleaning script
├── data_preprocess.py      # Data preprocessing script
└── README.md               # Instructions file
```

---

## ⚙️ Requirements

- Python 3.8 or higher  
- Recommended libraries:
  ```bash
  pip install pandas numpy scikit-learn
  ```

---

## 🚀 How to Run

### 1. Clone or download the project
```bash
git clone <your_repo_url>
cd <your_project_folder>
```

### 2. Clean the dataset
Run the data cleaning script with your CSV file:
```bash
python data_cleaning.py dataset_1.csv
```

This script will:
- Remove duplicates  
- Handle missing values  
- Save the cleaned file (e.g., `cleaned_dataset.csv`)

### 3. Preprocess the cleaned dataset
Next, run the preprocessing script:
```bash
python data_preprocess.py cleaned_dataset.csv
```

This step typically:
- Encodes categorical variables  
- Normalizes numeric columns  
- Splits data into training and testing sets  
- Outputs the final processed dataset (e.g., `preprocessed_dataset.csv`)

---

## 🧠 Example Output

After successful execution, you should have:
```
cleaned_dataset.csv
preprocessed_dataset.csv
```

---

## 🛠️ Customization

- To modify data cleaning rules, open `data_cleaning.py` and adjust the cleaning logic.
- To change preprocessing parameters (e.g., scaling method or train/test split), edit `data_preprocess.py`.

---

## 🧾 License

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).

---

## 👤 Author

Created by [Your Name]  
📧 Contact: your.email@example.com
