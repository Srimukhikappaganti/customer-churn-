from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from data_utils import load_data, prepare_model_frame

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'


def main(n_topics=6, top_words=10):
    df = prepare_model_frame(load_data())
    texts = df['customer_text_pre_outcome'].fillna('')
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2), min_df=3, max_df=0.95, max_features=5000)
    X = vec.fit_transform(texts)
    nmf = NMF(n_components=n_topics, random_state=42, init='nndsvda', max_iter=500)
    W = nmf.fit_transform(X)
    terms = vec.get_feature_names_out()
    rows=[]
    for i, topic in enumerate(nmf.components_):
        words = [terms[j] for j in topic.argsort()[-top_words:][::-1]]
        rows.append({'topic': i+1, 'top_terms': ', '.join(words)})
    topic_df = pd.DataFrame(rows)
    topic_df.to_csv(REPORTS / 'topics.csv', index=False)
    for i in range(n_topics):
        df[f'topic_{i+1}_score'] = W[:, i]
    df['churned'] = df['target_churn']
    summary=[]
    for i in range(n_topics):
        c=f'topic_{i+1}_score'
        summary.append({'topic': i+1, 'mean_score_retained': df.loc[df.churned==0,c].mean(), 'mean_score_churned': df.loc[df.churned==1,c].mean()})
    pd.DataFrame(summary).to_csv(REPORTS / 'topic_churn_comparison.csv', index=False)
    print(topic_df.to_string(index=False))
    print('\nTopic/churn comparison:')
    print(pd.DataFrame(summary).to_string(index=False))

if __name__ == '__main__':
    main()
