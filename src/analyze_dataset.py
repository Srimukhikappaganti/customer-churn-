from pathlib import Path
import json
import pandas as pd
from data_utils import load_data

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
REPORTS.mkdir(exist_ok=True)

def main():
    df = load_data()
    report = {
        'rows': int(len(df)),
        'columns': int(df.shape[1]),
        'duplicate_rows': int(df.drop(columns=['conversation','participants','churn_signals'], errors='ignore').duplicated().sum()),
        'duplicate_conversation_id': int(df['conversation_id'].duplicated().sum()),
        'missing_values': {k: int(v) for k, v in df.isna().sum().items() if v},
        'churn_risk_level': df['churn_risk_level'].value_counts().to_dict(),
        'resolution_outcome': df['resolution_outcome'].value_counts().to_dict(),
        'target_churn_count': int(df['resolution_outcome'].astype(str).str.startswith('churned').sum()),
        'target_churn_rate': float(df['resolution_outcome'].astype(str).str.startswith('churned').mean()),
        'columns': list(df.columns),
        'categorical_cardinality': {c: int(df[c].astype(str).nunique(dropna=False)) for c in df.select_dtypes(include=['object']).columns if c not in ['conversation','participants','churn_signals','summary']}
    }
    (REPORTS / 'dataset_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    pd.DataFrame({'column': df.columns, 'dtype': [str(df[c].dtype) for c in df.columns], 'missing': [int(df[c].isna().sum()) for c in df.columns], 'unique': [int(df[c].astype(str).nunique(dropna=False)) for c in df.columns]}).to_csv(REPORTS / 'column_profile.csv', index=False)
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
