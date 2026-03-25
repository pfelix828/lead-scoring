"""
Lead Scoring & Buying Group Dashboard

Interactive exploration of propensity scores and buying group coverage gaps.
Runs entirely on local synthetic data — no APIs required.
"""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

from src.generate_data import generate_all
from src.features import get_modeling_dataset, build_account_features
from src.model import train_logistic_regression, get_feature_importance
from src.buying_groups import (
    score_buying_group_completeness,
    identify_coverage_gaps,
    estimate_enrichment_pipeline,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Lead Scoring & Buying Groups",
    page_icon="🎯",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Data loading & model training (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    # Generate data in-memory (works both locally and on Streamlit Cloud
    # where the gitignored CSVs don't exist)
    tables = generate_all()
    return {
        "accounts": tables["accounts"],
        "contacts": tables["contacts"],
        "opportunities": tables["opportunities"],
        "contact_opp": tables["contact_opportunity"],
    }


@st.cache_data
def build_scored_dataset(_data):
    accounts = _data["accounts"]
    contacts = _data["contacts"]
    opportunities = _data["opportunities"]
    contact_opp = _data["contact_opp"]

    # Build features and train model
    X, y = get_modeling_dataset(accounts, contacts, opportunities, contact_opp)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    lr = train_logistic_regression(X_train, y_train, cv=5)

    # Score all accounts and track train/test membership
    scores = lr["model"].predict_proba(X)[:, 1]
    in_test_set = pd.Series(False, index=X.index)
    in_test_set.loc[X_test.index] = True

    # Rebuild with account IDs
    acct_feat = build_account_features(accounts, contacts, opportunities, contact_opp)
    closed = opportunities[opportunities["stage"].isin(["Closed Won", "Closed Lost"])]
    acct_target = closed.groupby("account_id")["is_won"].max().reset_index()
    acct_target.columns = ["account_id", "target"]
    dataset = acct_feat.merge(acct_target, on="account_id", how="inner")
    dataset["propensity_score"] = scores
    dataset["in_test_set"] = in_test_set.values

    # Buying group completeness
    completeness = score_buying_group_completeness(
        accounts, contacts, contact_opp, opportunities
    )

    # Merge everything
    result = dataset[["account_id", "propensity_score", "target", "in_test_set"]].merge(
        completeness, on="account_id", how="inner"
    ).merge(
        accounts[["account_id", "company_name", "segment", "industry",
                  "employee_count", "annual_revenue", "region", "tech_stack",
                  "has_existing_product", "arr"]],
        on="account_id", how="left"
    )

    # Feature importance
    importance = get_feature_importance(lr["model"], list(X.columns))

    # Model metrics
    from src.model import evaluate_model
    eval_result = evaluate_model(lr["model"], X_test, y_test)

    return result, importance, eval_result, lr["cv_auc"]


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------
data = load_data()
scored, importance, eval_result, cv_auc = build_scored_dataset(data)

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.title("Filters")

segments = st.sidebar.multiselect(
    "Segment",
    options=sorted(scored["segment"].unique()),
    default=sorted(scored["segment"].unique()),
)
industries = st.sidebar.multiselect(
    "Industry",
    options=sorted(scored["industry"].unique()),
    default=sorted(scored["industry"].unique()),
)
score_range = st.sidebar.slider(
    "Propensity Score Range",
    min_value=0.0,
    max_value=1.0,
    value=(0.0, 1.0),
    step=0.05,
)
completeness_range = st.sidebar.slider(
    "Completeness Score Range",
    min_value=0,
    max_value=100,
    value=(0, 100),
    step=5,
)
test_set_only = st.sidebar.checkbox(
    "Test set accounts only (out-of-sample)",
    value=True,
    help="When checked, shows only accounts the model did NOT train on. "
         "Uncheck to include all accounts (training + test).",
)

# Apply filters
filtered = scored[
    (scored["segment"].isin(segments))
    & (scored["industry"].isin(industries))
    & (scored["propensity_score"].between(*score_range))
    & (scored["completeness_score"].between(*completeness_range))
]
if test_set_only:
    filtered = filtered[filtered["in_test_set"]]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Lead Scoring & Buying Group Dashboard")
st.markdown("Propensity model and buying group analysis for B2B SaaS account targeting.")

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accounts", f"{len(filtered):,}")
col2.metric("Win Rate", f"{filtered['target'].mean():.0%}")
col3.metric("Test AUC", f"{eval_result['metrics']['auc']:.3f}")
col4.metric("Precision @ 10%", f"{eval_result['metrics']['precision_at_10pct']:.0%}")
col5.metric("Median Completeness", f"{filtered['completeness_score'].median():.0f}")

st.divider()

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Propensity Scores", "Buying Groups", "Gap Analysis", "Model Details"
])

# ---------------------------------------------------------------------------
# Tab 1: Propensity Scores
# ---------------------------------------------------------------------------
with tab1:
    left, right = st.columns(2)

    with left:
        st.subheader("Score Distribution by Outcome")
        fig = go.Figure()
        won = filtered[filtered["target"] == 1]["propensity_score"]
        lost = filtered[filtered["target"] == 0]["propensity_score"]
        fig.add_trace(go.Histogram(x=lost, name="Lost", marker_color="#e74c3c",
                                   opacity=0.6, nbinsx=25))
        fig.add_trace(go.Histogram(x=won, name="Won", marker_color="#2ecc71",
                                   opacity=0.6, nbinsx=25))
        fig.update_layout(barmode="overlay", xaxis_title="Propensity Score",
                         yaxis_title="Accounts", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Win Rate by Score Tier")
        tier_bins = [0, 0.25, 0.35, 0.45, 1.0]
        tier_labels = ["Low", "Medium", "High", "Very High"]
        temp = filtered.copy()
        temp["tier"] = pd.cut(temp["propensity_score"], bins=tier_bins, labels=tier_labels)
        tier_stats = temp.groupby("tier", observed=False).agg(
            win_rate=("target", "mean"),
            count=("target", "count"),
        ).reset_index()
        fig = px.bar(tier_stats, x="tier", y="win_rate", text="count",
                     color="win_rate", color_continuous_scale="Greens",
                     labels={"tier": "Score Tier", "win_rate": "Win Rate", "count": "Accounts"})
        fig.update_traces(texttemplate="n=%{text}", textposition="outside")
        fig.update_layout(yaxis_tickformat=".0%", height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Top accounts table
    st.subheader("Top Scored Accounts")
    top = filtered.nlargest(20, "propensity_score")[
        ["company_name", "segment", "industry", "propensity_score",
         "completeness_score", "target"]
    ].copy()
    top["propensity_score"] = top["propensity_score"].apply(lambda x: f"{x:.1%}")
    top["completeness_score"] = top["completeness_score"].apply(lambda x: f"{x:.0f}")
    top["target"] = top["target"].map({1: "Won", 0: "Lost"})
    top.columns = ["Company", "Segment", "Industry", "Propensity", "BG Completeness", "Outcome"]
    st.dataframe(top, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Tab 2: Buying Groups
# ---------------------------------------------------------------------------
with tab2:
    left, right = st.columns(2)

    with left:
        st.subheader("Completeness vs. Win Rate")
        temp = filtered[filtered["contact_count"] > 0].copy()
        temp["tier"] = pd.cut(
            temp["completeness_score"],
            bins=[-1, 25, 50, 75, 100],
            labels=["Low (0-25)", "Medium (25-50)", "High (50-75)", "Complete (75-100)"],
        )
        tier_data = temp.groupby("tier", observed=False).agg(
            win_rate=("target", "mean"),
            count=("target", "count"),
        ).reset_index()
        fig = px.bar(tier_data, x="tier", y="win_rate", text="count",
                     color="win_rate", color_continuous_scale="Blues",
                     labels={"tier": "Completeness Tier", "win_rate": "Win Rate"})
        fig.update_traces(texttemplate="n=%{text}", textposition="outside")
        fig.update_layout(yaxis_tickformat=".0%", height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Propensity × Completeness")
        fig = px.scatter(
            filtered[filtered["contact_count"] > 0],
            x="completeness_score",
            y="propensity_score",
            color=filtered[filtered["contact_count"] > 0]["target"].map({1: "Won", 0: "Lost"}),
            color_discrete_map={"Won": "#2ecc71", "Lost": "#e74c3c"},
            opacity=0.5,
            hover_data=["company_name", "segment", "industry"],
            labels={"completeness_score": "BG Completeness", "propensity_score": "Propensity Score",
                    "color": "Outcome"},
        )
        fig.add_hline(y=filtered["propensity_score"].median(), line_dash="dash",
                      line_color="gray", opacity=0.5)
        fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Sub-score breakdown by segment
    st.subheader("Average Sub-Scores by Segment")
    sub_cols = ["role_coverage_score", "seniority_mix_score",
                "function_diversity_score", "tech_business_score"]
    seg_sub = filtered[filtered["contact_count"] > 0].groupby("segment")[sub_cols].mean()
    seg_sub.columns = ["Role Coverage", "Seniority Mix", "Function Diversity", "Tech + Business"]
    seg_sub = seg_sub.reindex(["SMB", "Mid-Market", "Enterprise"])

    fig = px.bar(seg_sub.reset_index(), x="segment", y=seg_sub.columns.tolist(),
                 barmode="group", labels={"value": "Avg Score (out of 25)", "segment": "Segment"},
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=400, yaxis_range=[0, 25])
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3: Gap Analysis
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Enrichment Targets")
    st.markdown(
        "Accounts with **high propensity but incomplete buying groups** — "
        "the model says they're likely to buy, they just need the right people at the table."
    )

    # Compute gaps on filtered data
    gap_accounts = filtered[
        (filtered["propensity_score"] >= filtered["propensity_score"].median())
        & (filtered["completeness_score"] < 50)
        & (filtered["contact_count"] > 0)
    ].sort_values("propensity_score", ascending=False)

    col1, col2, col3 = st.columns(3)
    col1.metric("Enrichment Targets", len(gap_accounts))
    col2.metric("Avg Propensity", f"{gap_accounts['propensity_score'].mean():.0%}" if len(gap_accounts) > 0 else "—")
    col3.metric("Avg Completeness", f"{gap_accounts['completeness_score'].mean():.0f}" if len(gap_accounts) > 0 else "—")

    if len(gap_accounts) > 0:
        # Common gaps
        from collections import Counter
        all_missing = []
        for roles in gap_accounts["roles_missing"]:
            if isinstance(roles, list):
                all_missing.extend(roles)
        role_counts = Counter(all_missing)

        if role_counts:
            left, right = st.columns(2)
            with left:
                st.subheader("Most Common Missing Roles")
                gap_df = pd.DataFrame(role_counts.items(), columns=["Role", "Accounts Missing"])
                gap_df = gap_df.sort_values("Accounts Missing", ascending=True)
                fig = px.bar(gap_df, x="Accounts Missing", y="Role", orientation="h",
                             color="Accounts Missing", color_continuous_scale="Reds")
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with right:
                st.subheader("Structural Gaps")
                struct_gaps = pd.DataFrame({
                    "Gap Type": ["No VP+ Contact", "No Technical Function", "No Business Function"],
                    "Accounts": [
                        (~gap_accounts["has_vp_plus"]).sum(),
                        (~gap_accounts["has_technical"]).sum(),
                        (~gap_accounts["has_business"]).sum(),
                    ],
                })
                fig = px.bar(struct_gaps, x="Gap Type", y="Accounts", color="Accounts",
                             color_continuous_scale="Oranges")
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # Target list
        st.subheader("Target Account List")
        display = gap_accounts[
            ["company_name", "segment", "industry", "propensity_score",
             "completeness_score", "roles_missing", "has_vp_plus"]
        ].head(30).copy()
        display["propensity_score"] = display["propensity_score"].apply(lambda x: f"{x:.1%}")
        display["completeness_score"] = display["completeness_score"].apply(lambda x: f"{x:.0f}")
        display["roles_missing"] = display["roles_missing"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) and x else "—"
        )
        display["has_vp_plus"] = display["has_vp_plus"].map({True: "Yes", False: "No"})
        display.columns = ["Company", "Segment", "Industry", "Propensity",
                          "Completeness", "Missing Roles", "VP+ Present"]
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.info("No enrichment targets match the current filters.")

# ---------------------------------------------------------------------------
# Tab 4: Model Details
# ---------------------------------------------------------------------------
with tab4:
    left, right = st.columns(2)

    with left:
        st.subheader("Feature Importance (Top 20)")
        top_imp = importance.head(20).copy()
        col_name = [c for c in top_imp.columns if c != "feature"][0]
        top_imp = top_imp.iloc[::-1]  # reverse for horizontal bar
        fig = px.bar(top_imp, x=col_name, y="feature", orientation="h",
                     color=top_imp[col_name].apply(lambda x: "Positive" if x > 0 else "Negative"),
                     color_discrete_map={"Positive": "#2ecc71", "Negative": "#e74c3c"},
                     labels={col_name: "Coefficient", "feature": ""})
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Model Performance")
        metrics = eval_result["metrics"]

        st.markdown(f"""
| Metric | Value |
|--------|-------|
| **Test AUC** | {metrics['auc']:.3f} |
| **CV AUC** | {cv_auc:.3f} |
| **Log Loss** | {metrics['log_loss']:.3f} |
| **Precision @ 10%** | {metrics['precision_at_10pct']:.1%} |
| **Precision @ 20%** | {metrics['precision_at_20pct']:.1%} |
| **Precision @ 30%** | {metrics['precision_at_30pct']:.1%} |
        """)

        st.subheader("Lift by Decile")
        lift = metrics["lift_by_decile"]
        fig = px.bar(lift, x="decile", y="lift",
                     text=lift["lift"].apply(lambda x: f"{x:.1f}x"),
                     labels={"decile": "Score Decile (1 = Highest)", "lift": "Lift vs Baseline"})
        fig.add_hline(y=1.0, line_dash="dash", line_color="red", opacity=0.7,
                      annotation_text="Baseline (1.0x)")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Cumulative Win Capture")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lift["decile"], y=lift["cumulative_capture"],
            mode="lines+markers+text",
            text=lift["cumulative_capture"].apply(lambda x: f"{x:.0%}"),
            textposition="top center",
            marker=dict(color="#4C72B0", size=8),
            line=dict(color="#4C72B0", width=2),
        ))
        fig.update_layout(
            xaxis_title="Score Decile (1 = Highest)",
            yaxis_title="Cumulative % of Wins Captured",
            yaxis_tickformat=".0%",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)
