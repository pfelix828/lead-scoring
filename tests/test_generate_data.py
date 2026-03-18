"""Tests for synthetic data generation."""

import pandas as pd
from src.generate_data import generate_all


class TestAccounts:
    def test_shape(self, sample_accounts):
        assert len(sample_accounts) == 30
        assert "account_id" in sample_accounts.columns

    def test_no_null_keys(self, sample_accounts):
        assert sample_accounts["account_id"].notna().all()
        assert sample_accounts["account_id"].is_unique

    def test_segments(self, sample_accounts):
        valid = {"SMB", "Mid-Market", "Enterprise"}
        assert set(sample_accounts["segment"].unique()).issubset(valid)

    def test_employee_count_range(self, sample_accounts):
        assert (sample_accounts["employee_count"] >= 20).all()
        assert (sample_accounts["employee_count"] <= 50000).all()

    def test_tech_stack_not_empty(self, sample_accounts):
        assert sample_accounts["tech_stack"].str.len().gt(0).all()

    def test_arr_zero_for_non_customers(self, sample_accounts):
        non_customers = sample_accounts[~sample_accounts["has_existing_product"]]
        if len(non_customers) > 0:
            assert (non_customers["arr"] == 0).all()


class TestContacts:
    def test_has_contacts(self, sample_contacts):
        assert len(sample_contacts) > 0

    def test_no_null_keys(self, sample_contacts):
        assert sample_contacts["contact_id"].notna().all()
        assert sample_contacts["contact_id"].is_unique

    def test_referential_integrity(self, sample_accounts, sample_contacts):
        valid_accounts = set(sample_accounts["account_id"])
        assert set(sample_contacts["account_id"]).issubset(valid_accounts)

    def test_seniority_values(self, sample_contacts):
        valid = {"C-Suite", "VP", "Director", "Manager", "Individual Contributor"}
        assert set(sample_contacts["seniority"].unique()).issubset(valid)

    def test_function_values(self, sample_contacts):
        valid = {"Engineering", "IT", "Marketing", "Sales", "Product",
                 "Finance", "Operations", "Executive"}
        assert set(sample_contacts["job_function"].unique()).issubset(valid)

    def test_min_contacts_per_account(self, sample_accounts, sample_contacts):
        counts = sample_contacts.groupby("account_id").size()
        assert (counts >= 2).all()


class TestOpportunities:
    def test_has_opportunities(self, sample_opps):
        assert len(sample_opps) > 0

    def test_no_null_keys(self, sample_opps):
        assert sample_opps["opportunity_id"].notna().all()
        assert sample_opps["opportunity_id"].is_unique

    def test_is_won_is_boolean(self, sample_opps):
        assert sample_opps["is_won"].dtype == bool

    def test_amount_positive(self, sample_opps):
        assert (sample_opps["amount"] > 0).all()

    def test_stages(self, sample_opps):
        valid = {"Discovery", "Evaluation", "Proposal", "Negotiation",
                 "Closed Won", "Closed Lost"}
        assert set(sample_opps["stage"].unique()).issubset(valid)

    def test_won_deals_are_closed_won(self, sample_opps):
        won = sample_opps[sample_opps["is_won"]]
        assert (won["stage"] == "Closed Won").all()


class TestBridgeTable:
    def test_has_rows(self, sample_bridge):
        assert len(sample_bridge) > 0

    def test_valid_roles(self, sample_bridge):
        valid = {"Champion", "Decision Maker", "Influencer", "Evaluator", "Blocker"}
        assert set(sample_bridge["role"].unique()).issubset(valid)

    def test_referential_integrity_contacts(self, sample_contacts, sample_bridge):
        valid = set(sample_contacts["contact_id"])
        assert set(sample_bridge["contact_id"]).issubset(valid)

    def test_referential_integrity_opps(self, sample_opps, sample_bridge):
        valid = set(sample_opps["opportunity_id"])
        assert set(sample_bridge["opportunity_id"]).issubset(valid)


class TestGenerateAll:
    def test_returns_four_tables(self):
        tables = generate_all()
        assert set(tables.keys()) == {"accounts", "contacts", "opportunities", "contact_opportunity"}
        for name, df in tables.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
