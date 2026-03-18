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
