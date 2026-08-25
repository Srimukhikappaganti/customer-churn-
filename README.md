# Customer Churn Intelligence Using NLP and Machine Learning

## Project overview

This portfolio project analyzes customer-support conversations to discover recurring issues that appear before customers churn and builds a machine-learning baseline for churn-risk scoring.

The project uses the **Customer-Churn-Dataset-V2** synthetic SaaS/customer-support dataset. The dataset contains 500 conversation records and includes conversation text, customer/account attributes, churn-related labels/signals, and resolution outcomes. The current profiling run found 0 duplicate rows, 0 duplicate conversation IDs, and one column with missing values: `discount_offered_pct` (396 missing). The `churn_risk_level` field contains 41 records labelled `churned`, while the outcome-based target used by this project contains 61 records whose `resolution_outcome` starts with `churned`. The two fields are therefore treated as different concepts.

> **Important:** The source dataset is synthetic. Results should be treated as a portfolio demonstration, not as production evidence.

## Business problem

Customer churn means a customer stops using a product or leaves a subscription. Support conversations often contain warning signs before that happens. The project asks:

1. What problems do customers discuss before churn?
2. Which conversation topics are more common among churned customers?
3. Which customer/account patterns are associated with churn?
4. Can a model estimate churn risk while avoiding obvious target leakage?

## Methodology

### 1. Data understanding
- Profile rows, columns, missing values, duplicates, categorical distributions, and churn outcomes.
- Keep the original `data/full.jsonl` unchanged.

### 2. NLP analysis
- Extract customer-only conversation text.
- Exclude the final customer message from the modeling text to reduce direct outcome leakage.
- Convert text to TF-IDF features.
- Use NMF topic modeling to surface recurring themes.
- Compare topic strength between churned and retained records.

### 3. Machine learning
The target is derived from `resolution_outcome`: outcomes beginning with `churned` are treated as churned.

The baseline uses:
- Logistic Regression with class balancing
- TF-IDF features from pre-outcome customer text
- One-hot encoded categorical account/context features
- Scaled numerical features
- Stratified train/test split
- 5-fold stratified ROC-AUC cross-validation

### Leakage controls
The following fields are deliberately excluded from model predictors because they can reveal the outcome or are identifiers/metadata artifacts:

- `churn_risk_level`
- `churn_signals`
- `resolution_outcome`
- `summary`
- `discount_offered_pct`
- `conversation_id`
- `participants`
- `quality_score`
- `stale_phrase_count`
- `injection_style`
- final customer message

## Project structure

```text
customer-churn-nlp/
├── data/
│   └── full.jsonl
├── dashboard/
│   └── app.py
├── models/
├── reports/
├── src/
│   ├── data_utils.py
│   ├── analyze_dataset.py
│   ├── topic_analysis.py
│   └── train_model.py
├── .gitignore
├── requirements.txt
└── README.md
```

## How to run

### 1. Create an environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Profile the dataset

```bash
python src/analyze_dataset.py
```

This creates a dataset report under `reports/`.

### 4. Discover conversation topics

```bash
python src/topic_analysis.py
```

Outputs:
- `reports/topics.csv`
- `reports/topic_churn_comparison.csv`

### 5. Train the churn model

```bash
python src/train_model.py
```

Outputs:
- `models/churn_pipeline.joblib`
- `reports/model_metrics.json`

Current baseline result from the supplied dataset (80/20 stratified holdout): ROC-AUC **0.969**. Five-fold stratified cross-validation ROC-AUC: **0.943 ± 0.016**. Because the dataset is synthetic and contains only 61 outcome-based churned records, these metrics should not be presented as production performance.

### 6. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

## Interview explanation

> I built a customer churn intelligence project using NLP and machine learning. I analyzed customer-support conversations to identify recurring issues associated with churn, used TF-IDF and NMF topic modeling to surface conversation themes, and trained a class-balanced logistic regression model using pre-outcome customer text and account features. I also explicitly controlled for target leakage by excluding churn labels, resolution outcomes, summaries, and other post-outcome or metadata fields from the predictors.

## Limitations

- The dataset is synthetic and small, with only 500 records.
- There are only 41 explicitly churned records, so model metrics can be unstable.
- The dataset does not provide a real multi-day customer timeline, so sequence claims are limited to conversation-level behavior.
- The model is a portfolio baseline, not a production churn system.
