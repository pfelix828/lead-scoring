"""Tests for feature engineering module."""

import numpy as np
import pandas as pd
from src.features import (
    build_contact_features,
    build_account_features,
    get_modeling_dataset,
)


class TestContactFeatures:
    def test_returns_dataframe(self, sample_accounts, sample_contacts):
        result = build_contact_features(sample_contacts, sample_accounts)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_contacts)

    def test_seniority_rank(self, sample_accounts, sample_contacts):
        result = build_contact_features(sample_contacts, sample_accounts)
        assert "seniority_rank" in result.columns
        assert result["seniority_rank"].between(1, 5).all()

    def test_binary_features(self, sample_accounts, sample_contacts):
        result = build_contact_features(sample_contacts, sample_accounts)
        for col in ["is_vp_plus", "is_director_plus", "is_technical", "is_business"]:
            assert col in result.columns
            assert result[col].isin([0, 1]).all()

    def test_function_dummies(self, sample_accounts, sample_contacts):
        result = build_contact_features(sample_contacts, sample_accounts)
        func_cols = [c for c in result.columns if c.startswith("func_")]
        assert len(func_cols) > 0
        for col in func_cols:
            assert result[col].isin([0, 1]).all()

    def test_account_features_merged(self, sample_accounts, sample_contacts):
        result = build_contact_features(sample_contacts, sample_accounts)
        assert "employee_count_log" in result.columns
        assert result["employee_count_log"].notna().all()


class TestAccountFeatures:
    def test_returns_dataframe(self, sample_accounts, sample_contacts, sample_opps, sample_bridge):
        result = build_account_features(
            sample_accounts, sample_contacts, sample_opps, sample_bridge
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_accounts)

    def test_no_nan_after_fill(self, sample_accounts, sample_contacts, sample_opps, sample_bridge):
        result = build_account_features(
            sample_accounts, sample_contacts, sample_opps, sample_bridge
        )
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert result[numeric_cols].notna().all().all()

    def test_log_features_non_negative(self, sample_accounts, sample_contacts, sample_opps, sample_bridge):
        result = build_account_features(
            sample_accounts, sample_contacts, sample_opps, sample_bridge
        )
        for col in ["employee_count_log", "annual_revenue_log"]:
            assert (result[col] >= 0).all()

    def test_contact_composition(self, sample_accounts, sample_contacts, sample_opps, sample_bridge):
        result = build_account_features(
            sample_accounts, sample_contacts, sample_opps, sample_bridge
        )
        assert "contact_count" in result.columns
        assert "function_diversity" in result.columns
        assert "senior_density" in result.columns

    def test_tech_features(self, sample_accounts, sample_contacts, sample_opps, sample_bridge):
        result = build_account_features(
            sample_accounts, sample_contacts, sample_opps, sample_bridge
        )
        assert "tech_stack_count" in result.columns
        assert "high_signal_tool_count" in result.columns
        assert "has_salesforce" in result.columns

    def test_include_deal_features_true(self, sample_accounts, sample_contacts, sample_opps, sample_bridge):
        result = build_account_features(
            sample_accounts, sample_contacts, sample_opps, sample_bridge,
            include_deal_features=True,
        )
        # In-deal features should be present
        assert "deal_contact_count" in result.columns
        assert "buying_group_completeness" in result.columns

    def test_exclude_deal_features(self, sample_accounts, sample_contacts, sample_opps, sample_bridge):
        result = build_account_features(
            sample_accounts, sample_contacts, sample_opps, sample_bridge,
            include_deal_features=False,
        )
        # In-deal features should NOT be present
        assert "deal_contact_count" not in result.columns
        assert "buying_group_completeness" not in result.columns
        assert "has_champion" not in result.columns
        # Pre-deal features should still be present
        assert "contact_count" in result.columns
        assert "tech_stack_count" in result.columns


class TestModelingDataset:
    def test_returns_X_y(self, sample_accounts, sample_contacts, sample_opps, sample_bridge):
        X, y = get_modeling_dataset(
            sample_accounts, sample_contacts, sample_opps, sample_bridge
        )
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)

    def test_target_is_binary(self, sample_accounts, sample_contacts, sample_opps, sample_bridge):
        _, y = get_modeling_dataset(
            sample_accounts, sample_contacts, sample_opps, sample_bridge
        )
        assert y.isin([0, 1]).all()

    def test_no_target_in_features(self, sample_accounts, sample_contacts, sample_opps, sample_bridge):
        X, _ = get_modeling_dataset(
            sample_accounts, sample_contacts, sample_opps, sample_bridge
        )
        assert "target" not in X.columns
        assert "account_id" not in X.columns

    def test_no_nan_in_features(self, sample_accounts, sample_contacts, sample_opps, sample_bridge):
        X, _ = get_modeling_dataset(
            sample_accounts, sample_contacts, sample_opps, sample_bridge
        )
        assert X.notna().all().all()
