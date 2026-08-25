from pathlib import Path
import sys
import json

import joblib
import pandas as pd
import streamlit as st


from src.data_utils import load_data, prepare_model_frame

ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 0.4rem 0 1.2rem 0;
    }

    .hero h1 {
        font-size: 2.6rem;
        margin-bottom: 0.2rem;
    }

    .hero p {
        font-size: 1.05rem;
        color: #64748b;
    }

    .insight {
        padding: 1rem 1.1rem;
        border-radius: 0.7rem;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA
# ============================================================

df = prepare_model_frame(load_data())

# Target variable:
# 1 = churned
# 0 = retained
df["churned"] = df["target_churn"]


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>Customer Churn Intelligence</h1>
        <p>
            NLP + Machine Learning analysis of customer-support conversations
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Portfolio project using a synthetic customer-support dataset. "
    "The churn-risk score is a demonstration and is not a production decision system."
)


# ============================================================
# KEY PERFORMANCE INDICATORS
# ============================================================

churn_count = int(df["churned"].sum())
churn_rate = float(df["churned"].mean() * 100)
average_turns = float(df["turn_count"].mean())

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Conversations",
    f"{len(df):,}"
)

c2.metric(
    "Churned",
    f"{churn_count:,}"
)

c3.metric(
    "Churn Rate",
    f"{churn_rate:.1f}%"
)

c4.metric(
    "Average Turns",
    f"{average_turns:.1f}"
)


# ============================================================
# BUSINESS OVERVIEW
# ============================================================

st.header("Business Overview")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# CHURN BY SUPPORT CHANNEL
# ------------------------------------------------------------

with col1:

    st.subheader("Churn by Support Channel")

    channel = (
        df.groupby("channel")["churned"]
        .agg(
            churned="sum",
            conversations="count"
        )
        .reset_index()
    )

    channel["churn_rate_pct"] = (
        channel["churned"]
        / channel["conversations"]
        * 100
    )

    channel = channel.sort_values(
        "churn_rate_pct",
        ascending=False
    )

    st.bar_chart(
        channel,
        x="channel",
        y="churn_rate_pct"
    )

    st.caption(
        "Churn rate = churned conversations ÷ total conversations "
        "within each support channel."
    )

    display_channel = channel.rename(
        columns={
            "channel": "Support Channel",
            "churned": "Churned",
            "conversations": "Conversations",
            "churn_rate_pct": "Churn Rate (%)",
        }
    )

    st.dataframe(
        display_channel.style.format(
            {
                "Churn Rate (%)": "{:.1f}"
            }
        ),
        hide_index=True,
        use_container_width=True,
    )


# ------------------------------------------------------------
# RESOLUTION OUTCOMES
# ------------------------------------------------------------

with col2:

    st.subheader("Resolution Outcomes")

    outcomes = (
        df["resolution_outcome"]
        .value_counts()
        .rename_axis("outcome")
        .reset_index(name="count")
    )

    st.bar_chart(
        outcomes,
        x="outcome",
        y="count"
    )

    st.caption(
        "Conversation outcomes used to define the churn target."
    )


# ============================================================
# NLP FINDINGS
# ============================================================

st.header("NLP Findings")

topic_file = ROOT / "reports" / "topic_churn_comparison.csv"
top_terms_file = ROOT / "reports" / "topics.csv"


if topic_file.exists() and top_terms_file.exists():

    topic_compare = pd.read_csv(topic_file)
    top_terms = pd.read_csv(top_terms_file)

    topic_compare["churn_delta"] = (
        topic_compare["mean_score_churned"]
        - topic_compare["mean_score_retained"]
    )

    topic_compare = topic_compare.sort_values(
        "churn_delta",
        ascending=False
    )

    left, right = st.columns(2)


    # --------------------------------------------------------
    # TOPICS ASSOCIATED WITH CHURN
    # --------------------------------------------------------

    with left:

        st.subheader(
            "Topics More Associated With Churn"
        )

        display_topics = topic_compare.copy()

        display_topics["topic"] = (
            "Topic "
            + display_topics["topic"].astype(str)
        )

        st.bar_chart(
            display_topics,
            x="topic",
            y="churn_delta"
        )

        st.caption(
            "Positive values indicate that the topic "
            "appears more strongly in churned conversations."
        )


    # --------------------------------------------------------
    # DISCOVERED TOPIC THEMES
    # --------------------------------------------------------

    with right:

        st.subheader(
            "Discovered Topic Themes"
        )

        st.dataframe(
            top_terms,
            hide_index=True,
            use_container_width=True
        )


    # --------------------------------------------------------
    # STRONGEST NLP SIGNAL
    # --------------------------------------------------------

    if not topic_compare.empty:

        strongest = topic_compare.iloc[0]

        strongest_topic = int(
            strongest["topic"]
        )

        strongest_terms = top_terms.loc[
            top_terms["topic"] == strongest_topic,
            "top_terms"
        ]

        if not strongest_terms.empty:

            st.markdown(
                f"""
                <div class="insight">
                    <b>Key NLP signal:</b>
                    Topic {strongest_topic}
                    has the largest positive
                    churn-vs-retained difference.

                    Representative terms include:
                    <i>{strongest_terms.iloc[0]}</i>
                </div>
                """,
                unsafe_allow_html=True,
            )

else:

    st.warning(
        "NLP topic reports were not found. "
        "Run the project pipeline again to regenerate them."
    )


# ============================================================
# MACHINE LEARNING PERFORMANCE
# ============================================================

st.header("Machine Learning Performance")

metrics_file = ROOT / "reports" / "model_metrics.json"


if metrics_file.exists():

    metrics = json.loads(
        metrics_file.read_text(
            encoding="utf-8"
        )
    )

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Holdout ROC-AUC",
        f"{metrics['roc_auc']:.3f}"
    )

    m2.metric(
        "5-Fold CV ROC-AUC",
        f"{metrics['cv_roc_auc_mean']:.3f}"
    )

    m3.metric(
        "CV Variability",
        f"±{metrics['cv_roc_auc_std']:.3f}"
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[
            "Actual Retained",
            "Actual Churned"
        ],
        columns=[
            "Predicted Retained",
            "Predicted Churned"
        ],
    )

    st.subheader(
        "Holdout Confusion Matrix"
    )

    st.dataframe(
        cm,
        use_container_width=False
    )

    st.caption(
        "Model evaluation on the held-out test set."
    )

else:

    st.warning(
        "Model metrics were not found. "
        "Run the project pipeline first."
    )


# ============================================================
# CUSTOMER CONVERSATION EXPLORER
# ============================================================

st.header(
    "Customer Conversation Explorer"
)

selected = st.selectbox(
    "Select a conversation",
    df["conversation_id"].tolist()
)

record = df[
    df["conversation_id"] == selected
].iloc[0]


# ------------------------------------------------------------
# CUSTOMER METADATA
# ------------------------------------------------------------

meta_cols = [
    "conversation_id",
    "channel",
    "product_category",
    "tenure_months",
    "churn_risk_level",
    "sentiment_arc",
    "resolution_outcome",
    "plan_type",
    "mrr_usd",
]

meta = {
    column: record[column]
    for column in meta_cols
}

st.dataframe(
    pd.DataFrame([meta]),
    hide_index=True,
    use_container_width=True
)


# ------------------------------------------------------------
# FULL CONVERSATION
# ------------------------------------------------------------

with st.expander(
    "View Full Conversation",
    expanded=False
):

    for message in record["conversation"]:

        if message.get("role") == "customer":
            who = "Customer"
        else:
            who = "Agent"

        speaker = message.get(
            "speaker",
            ""
        )

        text = message.get(
            "text",
            ""
        )

        st.markdown(
            f"**{who} — {speaker}**"
        )

        st.write(text)


# ============================================================
# CHURN RISK SCORING DEMO
# ============================================================

model_path = (
    ROOT
    / "models"
    / "churn_pipeline.joblib"
)


if model_path.exists():

    st.header(
        "Churn Risk Scoring Demo"
    )

    st.info(
        "This score is a portfolio demonstration. "
        "It is trained on the synthetic dataset and "
        "should not be used as a production risk score."
    )


    # --------------------------------------------------------
    # FEATURES EXPECTED BY THE TRAINED MODEL
    # --------------------------------------------------------

    score_columns = [

        "customer_text_pre_outcome",

        "channel",

        "product_category",

        "customer_tenure",

        "sentiment_arc",

        "customer_persona",

        "company_size",

        "plan_type",

        "tenure_months",

        "seats",

        "active_seats",

        "per_seat_price_usd",

        "mrr_usd",

        "turn_count",

        "word_count",

        "seat_utilization",
    ]


    score_df = df[
        score_columns
    ].copy()


    # --------------------------------------------------------
    # SELECT CUSTOMER
    # --------------------------------------------------------

    chosen = st.selectbox(
        "Customer Record for Scoring",
        df["conversation_id"],
        key="score"
    )


    # Find selected record
    idx = df.index[
        df["conversation_id"] == chosen
    ][0]


    # --------------------------------------------------------
    # LOAD TRAINED MODEL
    # --------------------------------------------------------

    pipeline = joblib.load(
        model_path
    )


    # --------------------------------------------------------
    # PREDICT CHURN PROBABILITY
    # --------------------------------------------------------

    risk = float(
        pipeline.predict_proba(
            score_df.loc[[idx]]
        )[:, 1][0]
    )


    # --------------------------------------------------------
    # DISPLAY SCORE
    # --------------------------------------------------------

    st.metric(
        "Estimated Churn Probability",
        f"{risk * 100:.1f}%"
    )


else:

    st.warning(
        "Trained churn model was not found. "
        "Run the project pipeline first."
    ) 