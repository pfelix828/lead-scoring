"""Build the account-level scored dataset the dashboard renders.

Extracted from the Streamlit app so the same code path serves two uses:
live computation from the raw tables, and offline precomputation of the
compact artifacts the deployed app loads (scripts/precompute_scored.py).
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.buying_groups import score_buying_group_completeness
from src.features import build_account_features, get_modeling_dataset
from src.model import (
    evaluate_model,
    get_feature_importance,
    train_logistic_regression,
)


def build_scored_artifacts(data: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict, float]:
    """Train the model and assemble everything the dashboard needs.

    Returns (scored_accounts, feature_importance, eval_result, cv_auc).
    """
    accounts = data["accounts"]
    contacts = data["contacts"]
    opportunities = data["opportunities"]
    contact_opp = data["contact_opp"]

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
    eval_result = evaluate_model(lr["model"], X_test, y_test)

    return result, importance, eval_result, lr["cv_auc"]
