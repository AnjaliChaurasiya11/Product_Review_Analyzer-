# 🧠 AI-Powered Product Review Intelligence Dashboard

An interactive **Streamlit** dashboard that performs **Aspect-Based Sentiment Analysis (ABSA)** on product reviews using NLP and Machine Learning.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Product Selector** | Choose from multiple products (iPhone, Samsung, MacBook, etc.) |
| **Multi-Review Aggregation** | Analyzes *all* reviews for a product, not just one |
| **Aspect Dashboard** | Visual cards showing sentiment % per aspect (Battery, Camera, Price, Performance) |
| **Smart Aspect Detection** | Synonym-aware keyword dictionary for robust aspect extraction |
| **Add Review** | Submit new reviews that persist and trigger real-time re-analysis |
| **Key Insights** | Auto-generated insights: best/worst aspect, totals |
| **Bar Chart** | Aspect sentiment distribution visualization |
| **Model Evaluation** | Accuracy, Precision, Recall & Confusion Matrix for both models |
| **Dual Models** | TF-IDF + Logistic Regression (primary) and Naive Bayes (comparison) |

---

## 📁 Project Structure

```
Product Analyser/
├── data/
│   ├── sample_reviews.csv        # Training data for ML models
│   └── product_reviews.json      # Per-product review storage
├── models/                       # Saved ML model pipelines
├── app.py                        # Streamlit dashboard (main entry point)
├── model.py                      # TF-IDF + LR / NB pipelines
├── preprocessing.py              # Text cleaning (tokenization, stopwords)
├── utils.py                      # Aspect dictionary & extraction logic
├── evaluation.py                 # Model evaluation metrics
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NLTK data (auto-handled on first run)

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 3. Train models (auto-handled on first run)

```bash
python model.py
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

---

## 🧠 NLP Pipeline

1. **Text Preprocessing** — Lowercasing, punctuation removal, NLTK tokenization, stopword removal
2. **Feature Extraction** — TF-IDF Vectorization
3. **Classification** — Logistic Regression (primary) and Multinomial Naive Bayes
4. **Aspect Detection** — Synonym-aware keyword matching from `utils.py`
5. **Aggregation** — Sentiment percentages computed per aspect across all reviews

---

## 📊 Evaluation

Both models are evaluated on the bundled `sample_reviews.csv` dataset:

- **Accuracy**
- **Precision** (weighted)
- **Recall** (weighted)
- **Confusion Matrix**

Results are displayed in the **Model Evaluation** tab of the dashboard.

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Streamlit** — Interactive UI
- **Scikit-learn** — ML models & TF-IDF
- **NLTK** — Tokenization & stopwords
- **Pandas** — Data handling

---

## 💬 Notes

- New reviews submitted via the sidebar are **persisted** to `data/product_reviews.json`.
- The dashboard re-analyzes all reviews in real-time on every interaction.
- Models auto-train from `data/sample_reviews.csv` if no saved models are found.
