import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'full.jsonl'


def load_data(path=DATA_PATH):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def customer_text(conversation, exclude_last_customer=True):
    if not isinstance(conversation, list):
        return ''
    msgs = [m for m in conversation if isinstance(m, dict) and m.get('role') == 'customer']
    if exclude_last_customer and len(msgs) > 1:
        msgs = msgs[:-1]
    return ' '.join(str(m.get('text', '')) for m in msgs).strip()


def prepare_model_frame(df):
    out = df.copy()
    out['target_churn'] = out['resolution_outcome'].astype(str).str.startswith('churned').astype(int)
    out['customer_text_pre_outcome'] = out['conversation'].apply(customer_text)
    out['seat_utilization'] = (out['active_seats'] / out['seats'].replace(0, pd.NA)).fillna(0).astype(float)
    return out
