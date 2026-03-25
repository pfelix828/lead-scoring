"""Tests for model training and evaluation."""

import numpy as np
import pandas as pd
import pytest
from src.features import get_modeling_dataset
from src.model import (
    train_logistic_regression,
    train_random_forest,
    evaluate_model,
    get_feature_importance,
    bootstrap_ci,
)


@pytest.fixture
def modeling_data(sample_accounts, sample_contacts, sample_opps, sample_bridge):
    X, y = get_modeling_dataset(sample_accounts, sample_contacts, sample_opps, sample_bridge)
    return X, y


class TestLogisticRegression:
    def test_trains_without_error(self, modeling_data):
        X, y = modeling_data
        if len(y.unique()) < 2:
            pytest.skip("Need both classes in target")
        result = train_logistic_regression(X, y, cv=2)
        assert "model" in result
        assert "cv_auc" in result

    def test_cv_auc_reasonable(self, modeling_data):
        X, y = modeling_data
        if len(y.unique()) < 2:
            pytest.skip("Need both classes in target")
        result = train_logistic_regression(X, y, cv=2)
        assert 0.0 <= result["cv_auc"] <= 1.0

    def test_predictions_in_range(self, modeling_data):
        X, y = modeling_data
        if len(y.unique()) < 2:
            pytest.skip("Need both classes in target")
        result = train_logistic_regression(X, y, cv=2)
        proba = result["model"].predict_proba(X)[:, 1]
        assert (proba >= 0).all() and (proba <= 1).all()


class TestRandomForest:
    def test_trains_without_error(self, modeling_data):
        X, y = modeling_data
        if len(y.unique()) < 2:
            pytest.skip("Need both classes in target")
        result = train_random_forest(X, y, cv=2)
        assert "model" in result


class TestEvaluation:
    def test_returns_metrics(self, modeling_data):
        X, y = modeling_data
        if len(y.unique()) < 2:
            pytest.skip("Need both classes in target")
        result = train_logistic_regression(X, y, cv=2)
        eval_result = evaluate_model(result["model"], X, y)
        assert "metrics" in eval_result
        assert "auc" in eval_result["metrics"]
        assert 0.0 <= eval_result["metrics"]["auc"] <= 1.0

    def test_lift_by_decile(self, modeling_data):
        X, y = modeling_data
        if len(y.unique()) < 2:
            pytest.skip("Need both classes in target")
        result = train_logistic_regression(X, y, cv=2)
        eval_result = evaluate_model(result["model"], X, y)
        lift = eval_result["metrics"]["lift_by_decile"]
        assert isinstance(lift, pd.DataFrame)
        assert "lift" in lift.columns


class TestBootstrapCI:
    def test_returns_point_and_bounds(self):
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0] * 10)
        y_scores = np.array([0.2, 0.3, 0.8, 0.7, 0.4, 0.6, 0.1, 0.9, 0.5, 0.35] * 10)
        from sklearn.metrics import roc_auc_score
        result = bootstrap_ci(y_true, y_scores, roc_auc_score, n_bootstrap=200)
        assert "point" in result
        assert "lower" in result
        assert "upper" in result
        assert result["lower"] <= result["point"] <= result["upper"]

    def test_ci_bounds_reasonable(self):
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0] * 10)
        y_scores = np.array([0.2, 0.3, 0.8, 0.7, 0.4, 0.6, 0.1, 0.9, 0.5, 0.35] * 10)
        from sklearn.metrics import roc_auc_score
        result = bootstrap_ci(y_true, y_scores, roc_auc_score, n_bootstrap=500)
        assert 0.0 <= result["lower"] <= 1.0
        assert 0.0 <= result["upper"] <= 1.0


class TestEvaluationCI:
    def test_auc_ci_in_metrics(self, modeling_data):
        X, y = modeling_data
        if len(y.unique()) < 2:
            pytest.skip("Need both classes in target")
        result = train_logistic_regression(X, y, cv=2)
        eval_result = evaluate_model(result["model"], X, y)
        assert "auc_ci_lower" in eval_result["metrics"]
        assert "auc_ci_upper" in eval_result["metrics"]
        assert eval_result["metrics"]["auc_ci_lower"] <= eval_result["metrics"]["auc"]
        assert eval_result["metrics"]["auc_ci_upper"] >= eval_result["metrics"]["auc"]


class TestFeatureImportance:
    def test_logistic_regression(self, modeling_data):
        X, y = modeling_data
        if len(y.unique()) < 2:
            pytest.skip("Need both classes in target")
        result = train_logistic_regression(X, y, cv=2)
        importance = get_feature_importance(result["model"], list(X.columns))
        assert isinstance(importance, pd.DataFrame)
        assert len(importance) == len(X.columns)
        assert "feature" in importance.columns
        assert "coefficient" in importance.columns

    def test_random_forest(self, modeling_data):
        X, y = modeling_data
        if len(y.unique()) < 2:
            pytest.skip("Need both classes in target")
        result = train_random_forest(X, y, cv=2)
        importance = get_feature_importance(result["model"], list(X.columns))
        assert "importance" in importance.columns
