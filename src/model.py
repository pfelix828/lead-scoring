"""
Model training and evaluation for lead scoring.

Primary model: Logistic Regression (interpretable, defensible)
Secondary model: Random Forest (comparison benchmark)

All evaluation is business-framed: lift charts, precision@K, revenue capture.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    log_loss,
    classification_report,
)


def train_logistic_regression(X_train, y_train, cv: int = 5) -> dict:
    """
    Train a logistic regression model with cross-validated regularization.

    Returns dict with model, best params, and CV scores.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    param_grid = {
        "model__C": [0.01, 0.1, 1.0, 10.0],
    }

    search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, return_train_score=True
    )
    search.fit(X_train, y_train)

    return {
        "model": search.best_estimator_,
        "best_params": search.best_params_,
        "cv_auc": search.best_score_,
        "cv_results": pd.DataFrame(search.cv_results_),
    }


def train_random_forest(X_train, y_train, cv: int = 5) -> dict:
    """Train a Random Forest model for comparison."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])

    param_grid = {
        "model__max_depth": [5, 10, None],
        "model__min_samples_leaf": [5, 10, 20],
    }

    search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, return_train_score=True
    )
    search.fit(X_train, y_train)

    return {
        "model": search.best_estimator_,
        "best_params": search.best_params_,
        "cv_auc": search.best_score_,
        "cv_results": pd.DataFrame(search.cv_results_),
    }


def bootstrap_ci(y_true, y_scores, metric_fn, n_bootstrap=1000, ci=0.95):
    """
    Compute bootstrap confidence intervals for a metric.

    Args:
        y_true: True labels (array-like)
        y_scores: Predicted scores (array-like)
        metric_fn: Callable(y_true, y_scores) -> float
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (default 0.95)

    Returns:
        dict with keys 'point', 'lower', 'upper'
    """
    rng = np.random.default_rng(42)
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n = len(y_true)

    point = metric_fn(y_true, y_scores)

    boot_values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_t = y_true[idx]
        y_s = y_scores[idx]
        # Skip if only one class in bootstrap sample
        if len(np.unique(y_t)) < 2:
            continue
        boot_values.append(metric_fn(y_t, y_s))

    if len(boot_values) == 0:
        return {"point": point, "lower": point, "upper": point}

    alpha = (1 - ci) / 2
    lower = np.percentile(boot_values, alpha * 100)
    upper = np.percentile(boot_values, (1 - alpha) * 100)

    return {"point": point, "lower": lower, "upper": upper}


def evaluate_model(model, X_test, y_test, amounts=None) -> dict:
    """
    Evaluate a trained model with both ML and business metrics.

    Args:
        model: Trained sklearn Pipeline
        X_test: Test features
        y_test: Test labels
        amounts: Optional deal amounts for revenue capture analysis

    Returns dict of metrics and prediction arrays.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc_ci = bootstrap_ci(y_test, y_pred_proba, roc_auc_score)

    metrics = {
        "auc": auc_ci["point"],
        "auc_ci_lower": auc_ci["lower"],
        "auc_ci_upper": auc_ci["upper"],
        "log_loss": log_loss(y_test, y_pred_proba),
    }

    # Lift by decile
    metrics["lift_by_decile"] = _compute_lift_by_decile(y_test, y_pred_proba)

    # Precision at K with bootstrap CIs
    for k_pct in [10, 20, 30]:
        prec = _precision_at_k(y_test, y_pred_proba, k_pct / 100)
        metrics[f"precision_at_{k_pct}pct"] = prec

    # Bootstrap CI for precision@10%
    def _prec_at_10(y_t, y_s):
        return _precision_at_k(pd.Series(y_t), y_s, 0.10)

    p10_ci = bootstrap_ci(y_test, y_pred_proba, _prec_at_10)
    metrics["precision_at_10pct_ci_lower"] = p10_ci["lower"]
    metrics["precision_at_10pct_ci_upper"] = p10_ci["upper"]

    # Revenue capture (if amounts provided)
    if amounts is not None:
        metrics["revenue_capture"] = _compute_revenue_capture(
            y_test, y_pred_proba, amounts
        )

    return {
        "metrics": metrics,
        "y_pred_proba": y_pred_proba,
        "y_pred": y_pred,
    }


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importances from a trained Pipeline.

    For logistic regression, returns coefficients.
    For random forest, returns feature importances.
    """
    estimator = model.named_steps["model"]

    if hasattr(estimator, "coef_"):
        importance = estimator.coef_[0]
        col_name = "coefficient"
    elif hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
        col_name = "importance"
    else:
        raise ValueError("Model has no feature importances")

    return (
        pd.DataFrame({"feature": feature_names, col_name: importance})
        .assign(abs_value=lambda df: df[col_name].abs())
        .sort_values("abs_value", ascending=False)
        .drop(columns="abs_value")
        .reset_index(drop=True)
    )


# --- Business Metrics ---

def _compute_lift_by_decile(y_true, y_scores, n_deciles: int = 10) -> pd.DataFrame:
    """Compute lift chart data by score decile."""
    df = pd.DataFrame({"y_true": y_true.values, "score": y_scores})
    df["decile"] = pd.qcut(df["score"], n_deciles, labels=False, duplicates="drop")
    df["decile"] = df["decile"].max() - df["decile"] + 1  # 1 = highest scores

    baseline_rate = y_true.mean()

    lift = (
        df.groupby("decile")
        .agg(
            count=("y_true", "count"),
            wins=("y_true", "sum"),
            win_rate=("y_true", "mean"),
            avg_score=("score", "mean"),
        )
        .reset_index()
    )
    lift["lift"] = lift["win_rate"] / baseline_rate
    lift["cumulative_wins"] = lift["wins"].cumsum()
    lift["cumulative_capture"] = lift["cumulative_wins"] / y_true.sum()

    return lift


def _precision_at_k(y_true, y_scores, k_fraction: float) -> float:
    """Precision in the top-k% of scored items."""
    n = len(y_true)
    k = max(1, int(n * k_fraction))
    top_k_idx = np.argsort(y_scores)[::-1][:k]
    return y_true.values[top_k_idx].mean()


def _compute_revenue_capture(y_true, y_scores, amounts) -> pd.DataFrame:
    """How much revenue is captured at each score threshold."""
    df = pd.DataFrame({
        "y_true": y_true.values,
        "score": y_scores,
        "amount": amounts.values,
    })
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["won_revenue"] = df["y_true"] * df["amount"]
    df["cumulative_revenue"] = df["won_revenue"].cumsum()
    df["pct_accounts_reviewed"] = (df.index + 1) / len(df)
    df["pct_revenue_captured"] = df["cumulative_revenue"] / df["won_revenue"].sum()
    return df


# --- Plotting ---

def plot_lift_chart(lift_df: pd.DataFrame, title: str = "Lift by Score Decile"):
    """Plot a lift chart with baseline reference."""
    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(lift_df["decile"], lift_df["lift"], color="#4C72B0", edgecolor="white")
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="Baseline (1.0x)")

    for bar, (_, row) in zip(bars, lift_df.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{row["lift"]:.1f}x', ha="center", fontsize=9)

    ax.set_xlabel("Score Decile (1 = Highest)")
    ax.set_ylabel("Lift vs. Baseline")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_scores, model_name: str = "Model"):
    """Plot ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, color="#4C72B0", linewidth=2, label=f"{model_name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15):
    """Plot top feature importances as horizontal bar chart."""
    col_name = [c for c in importance_df.columns if c != "feature"][0]
    top = importance_df.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in top[col_name]]
    ax.barh(top["feature"], top[col_name], color=colors)
    ax.set_xlabel(col_name.title())
    ax.set_title(f"Top {top_n} Features")
    ax.axvline(x=0, color="black", linewidth=0.5)
    plt.tight_layout()
    return fig


def plot_revenue_capture(revenue_df: pd.DataFrame):
    """Plot cumulative revenue capture curve."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(revenue_df["pct_accounts_reviewed"] * 100,
            revenue_df["pct_revenue_captured"] * 100,
            color="#4C72B0", linewidth=2, label="Model")
    ax.plot([0, 100], [0, 100], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("% of Accounts Reviewed (Highest Score First)")
    ax.set_ylabel("% of Won Revenue Captured")
    ax.set_title("Revenue Capture Curve")
    ax.legend()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    return fig


def plot_calibration(y_true, y_scores, n_bins: int = 10):
    """Plot calibration curve: predicted vs. actual win rates."""
    df = pd.DataFrame({"y_true": y_true.values, "score": y_scores})
    df["bin"] = pd.qcut(df["score"], n_bins, labels=False, duplicates="drop")
    cal = df.groupby("bin").agg(
        predicted=("score", "mean"),
        actual=("y_true", "mean"),
        count=("y_true", "count"),
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(cal["predicted"], cal["actual"], "o-", color="#4C72B0", linewidth=2, markersize=8)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly Calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Win Rate")
    ax.set_title("Calibration Plot")
    ax.legend()
    plt.tight_layout()
    return fig
