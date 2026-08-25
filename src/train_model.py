from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from data_utils import load_data, prepare_model_frame

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'models'; REPORTS = ROOT / 'reports'
MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)

TEXT = 'customer_text_pre_outcome'
CATEGORICAL = ['channel','product_category','customer_tenure','sentiment_arc','customer_persona','company_size','plan_type']
NUMERIC = ['tenure_months','seats','active_seats','per_seat_price_usd','mrr_usd','turn_count','word_count','seat_utilization']


def build_pipeline():
    preprocess = ColumnTransformer([
        ('text', TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=2, max_features=5000, sublinear_tf=True), TEXT),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), CATEGORICAL),
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), NUMERIC),
    ])
    return Pipeline([('preprocess', preprocess), ('model', LogisticRegression(max_iter=3000, class_weight='balanced', random_state=42))])


def main():
    df = prepare_model_frame(load_data())
    X = df[[TEXT] + CATEGORICAL + NUMERIC]
    y = df['target_churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:,1]
    metrics = {
        'test_size': len(y_test),
        'roc_auc': float(roc_auc_score(y_test, proba)),
        'classification_report': classification_report(y_test, pred, output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, pred).tolist(),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(build_pipeline(), X, y, cv=cv, scoring='roc_auc')
    metrics['cv_roc_auc_mean'] = float(cv_auc.mean())
    metrics['cv_roc_auc_std'] = float(cv_auc.std())
    (REPORTS/'model_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    joblib.dump(pipe, MODELS/'churn_pipeline.joblib')
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
