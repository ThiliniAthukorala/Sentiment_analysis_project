# 📊 Sentiment Analysis Project

A machine learning project that classifies tweets/text as **Positive** or **Negative** sentiment using classical NLP preprocessing techniques and multiple ML classifiers.

---

## 📁 Project Structure

```
Sentiment_analysis_project/
│
├── notebooks/
│   ├── ModelBulding.ipynb        # Full ML pipeline: EDA, preprocessing, feature engineering, model training & evaluation
│   ├── Prediction_Pipline.ipynb  # Inference pipeline for new text input
│   └── download_dataset.ipynb    # Dataset download script
│
├── atrtifacts/
│   └── sentiment_analysis.csv    # Raw dataset (7,920 labeled tweets)
│
├── static/
│   └── model/
│       ├── model.pickle          # Serialized trained model (Logistic Regression)
│       ├── vocabulary.txt        # Filtered vocabulary tokens used for vectorization
│       └── corpora/
│           └── stopwords/
│               └── english       # NLTK English stopwords
│
├── requirements.txt              # Python dependencies
└── README.md
```

---

## 📦 Dataset

| Property        | Details                        |
|----------------|-------------------------------|
| **Source**      | Twitter / Social Media tweets  |
| **Size**        | 7,920 records                  |
| **Columns**     | `id`, `label`, `tweet`         |
| **Labels**      | `0` = Non-negative, `1` = Negative (hate/offensive) |
| **Duplicates**  | None                           |
| **Missing Values** | None                        |

---

## 🔧 Text Preprocessing Steps

All preprocessing is applied to the raw tweet text in the following order:

| Step | Description |
|------|-------------|
| **1. Lowercasing** | Converts all text to lowercase |
| **2. URL Removal** | Removes all HTTP/HTTPS hyperlinks using regex |
| **3. Punctuation Removal** | Strips all special characters and punctuation marks |
| **4. Number Removal** | Removes all numeric digits using regex (`\d+`) |
| **5. Stopword Removal** | Removes common English stopwords (NLTK corpus) |
| **6. Stemming** | Reduces words to their root form using **Porter Stemmer** (NLTK) |

---

## ⚙️ Feature Engineering

### Vocabulary Building
- A word **frequency counter** (`collections.Counter`) is built over all preprocessed tweets.
- Only tokens appearing **more than 10 times** are retained → resulting vocabulary size: **1,840 tokens**.
- The vocabulary is saved to `static/model/vocabulary.txt` for use during inference.

### Vectorization (Bag-of-Words)
- A **custom binary bag-of-words vectorizer** is implemented:
  - Each tweet is represented as a **binary vector of length 1,840**
  - A feature is set to `1` if the token is present in the tweet, `0` otherwise
- Training matrix shape: `(6,336, 1,840)`
- Test matrix shape: `(1,584, 1,840)`

### Handling Class Imbalance — SMOTE
- The training set is imbalanced (`Positive: 4,716` vs `Negative: 1,620`)
- **SMOTE (Synthetic Minority Oversampling Technique)** from `imbalanced-learn` is applied to balance the classes
- After SMOTE: `Positive: 4,716`, `Negative: 4,716` → Total training samples: **9,432**

### Train-Test Split
- **Split ratio**: 80% Train / 20% Test (`sklearn.model_selection.train_test_split`)

---

## 🤖 Models Trained & Evaluation Results

All models are trained on the SMOTE-balanced training set and evaluated on the original (unbalanced) test set.

### 1. 🔵 Logistic Regression ✅ *(Selected Model)*

| Metric     | Training | Testing |
|-----------|----------|---------|
| Accuracy   | 0.949    | **0.867** |
| Precision  | 0.923    | 0.698   |
| Recall     | 0.980    | 0.852   |
| F1-Score   | 0.951    | 0.767   |

> ✅ **Saved as the production model** (`static/model/model.pickle`)

---

### 2. 🟢 Multinomial Naive Bayes

| Metric     | Training | Testing |
|-----------|----------|---------|
| Accuracy   | 0.913    | 0.866   |
| Precision  | 0.874    | 0.671   |
| Recall     | 0.965    | 0.938   |
| F1-Score   | 0.917    | 0.782   |

---

### 3. 🟡 Decision Tree Classifier

| Metric     | Training | Testing |
|-----------|----------|---------|
| Accuracy   | 1.000    | 0.829   |
| Precision  | 1.000    | 0.681   |
| Recall     | 0.999    | 0.626   |
| F1-Score   | 1.000    | 0.652   |

> ⚠️ Overfitting observed — perfect training scores but lower test performance.

---

### 4. 🟠 Random Forest Classifier

| Metric     | Training | Testing |
|-----------|----------|---------|
| Accuracy   | 1.000    | 0.872   |
| Precision  | 1.000    | 0.777   |
| Recall     | 1.000    | 0.702   |
| F1-Score   | 1.000    | 0.737   |

> ⚠️ Overfitting observed on training data.

---

### 5. 🔴 Support Vector Machine (SVC)

| Metric     | Training | Testing |
|-----------|----------|---------|
| Accuracy   | 0.981    | **0.883** |
| Precision  | 0.966    | 0.746   |
| Recall     | 0.997    | 0.825   |
| F1-Score   | 0.981    | 0.784   |

> 💡 Highest test accuracy overall, but Logistic Regression was chosen for its speed and simplicity.

---

## 🔮 Inference / Prediction Pipeline

The `Prediction_Pipline.ipynb` applies the same preprocessing steps to new input text:

1. Load raw text input
2. Apply: lowercasing → URL removal → punctuation removal → number removal → stopword removal → stemming
3. Load vocabulary from `vocabulary.txt`
4. Vectorize using the same custom binary bag-of-words function
5. Load serialized model from `model.pickle`
6. Predict sentiment label (`0` = Positive/Non-negative, `1` = Negative)

---

## 🛠️ Technologies & Libraries

| Category              | Technology / Library              |
|----------------------|----------------------------------|
| **Language**          | Python 3.13                      |
| **Data Processing**   | `pandas`, `numpy`                |
| **Visualization**     | `matplotlib`                     |
| **NLP**               | `nltk` (stopwords, Porter Stemmer) |
| **ML Models**         | `scikit-learn` (v1.9)            |
| **Class Balancing**   | `imbalanced-learn` (SMOTE)       |
| **Model Serialization**| `pickle`                        |
| **Development Environment** | Jupyter Notebook          |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Sentiment_analysis_project
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. Run Notebooks
Open Jupyter Notebook and run in order:
1. `download_dataset.ipynb` — Download the dataset
2. `ModelBulding.ipynb` — Preprocess, train, and evaluate models
3. `Prediction_Pipline.ipynb` — Run predictions on new text

---

## 📈 Model Summary

```
Best Test Accuracy: SVM      → 88.3%
Chosen Production Model: Logistic Regression → 86.7% accuracy
Reason: Balance of accuracy, speed, and interpretability
```

---

## 📝 Notes

- The vocabulary is filtered to words occurring **> 10 times** to reduce noise and dimensionality.
- SMOTE is applied **only on training data** to avoid data leakage into the test set.
- The dataset contains social media text (tweets), so some preprocessing steps (URL removal, hashtag cleaning via punctuation stripping) are specifically tailored for this domain.
