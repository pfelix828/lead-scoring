"""Tests for buying group formation and gap analysis."""

import pandas as pd
from src.buying_groups import (
    score_buying_group_completeness,
    identify_coverage_gaps,
    completeness_vs_win_rate,
    estimate_enrichment_pipeline,
)


class TestCompleteness:
    def test_returns_all_accounts(self, sample_accounts, sample_contacts, sample_opps, sample_bridge):
        result = score_buying_group_completeness(
            sample_accounts, sample_contacts, sample_bridge, sample_opps
        )
        assert len(result) == len(sample_accounts)

    def test_score_range(self, sample_accounts, sample_contacts, sample_opps, sample_bridge):
        result = score_buying_group_completeness(
            sample_accounts, sample_contacts, sample_bridge, sample_opps
        )
        assert (result["completeness_score"] >= 0).all()
        assert (result["completeness_score"] <= 100).all()

    def test_sub_scores_sum_to_total(self, sample_accounts, sample_contacts, sample_opps, sample_bridge):
        result = score_buying_group_completeness(
            sample_accounts, sample_contacts, sample_bridge, sample_opps
        )
        sub_total = (
            result["role_coverage_score"]
            + result["seniority_mix_score"]
            + result["function_diversity_score"]
            + result["tech_business_score"]
        )
        assert (abs(result["completeness_score"] - sub_total) < 0.01).all()

    def test_no_deal_accounts_score_zero(self, sample_accounts, sample_contacts, sample_opps, sample_bridge):
        result = score_buying_group_completeness(
            sample_accounts, sample_contacts, sample_bridge, sample_opps
        )
        no_deals = result[result["contact_count"] == 0]
        if len(no_deals) > 0:
            assert (no_deals["completeness_score"] == 0).all()

    def test_roles_missing_is_list(self, sample_accounts, sample_contacts, sample_opps, sample_bridge):
        result = score_buying_group_completeness(
            sample_accounts, sample_contacts, sample_bridge, sample_opps
        )
        assert all(isinstance(r, list) for r in result["roles_missing"])


class TestCoverageGaps:
    def test_returns_dataframe(self, sample_accounts, sample_contacts, sample_opps, sample_bridge):
        completeness = score_buying_group_completeness(
            sample_accounts, sample_contacts, sample_bridge, sample_opps
        )
        gaps = identify_coverage_gaps(completeness, sample_accounts)
        assert isinstance(gaps, pd.DataFrame)

    def test_gaps_within_range(self, sample_accounts, sample_contacts, sample_opps, sample_bridge):
        completeness = score_buying_group_completeness(
            sample_accounts, sample_contacts, sample_bridge, sample_opps
        )
        gaps = identify_coverage_gaps(completeness, sample_accounts, min_completeness=25, max_completeness=75)
        if len(gaps) > 0:
            assert (gaps["completeness_score"] >= 25).all()
            assert (gaps["completeness_score"] <= 75).all()

    def test_has_enrichment_recommendation(self, sample_accounts, sample_contacts, sample_opps, sample_bridge):
        completeness = score_buying_group_completeness(
            sample_accounts, sample_contacts, sample_bridge, sample_opps
        )
        gaps = identify_coverage_gaps(completeness, sample_accounts)
        if len(gaps) > 0:
            assert "enrichment_recommendation" in gaps.columns
            assert gaps["enrichment_recommendation"].notna().all()


class TestCompletenessVsWinRate:
    def test_returns_tiers(self, sample_accounts, sample_contacts, sample_opps, sample_bridge):
        completeness = score_buying_group_completeness(
            sample_accounts, sample_contacts, sample_bridge, sample_opps
        )
        result = completeness_vs_win_rate(completeness, sample_opps)
        assert isinstance(result, pd.DataFrame)
        assert "completeness_tier" in result.columns
        assert "win_rate" in result.columns


class TestEnrichmentPipeline:
    def test_estimates_pipeline(self, sample_accounts, sample_contacts, sample_opps, sample_bridge):
        completeness = score_buying_group_completeness(
            sample_accounts, sample_contacts, sample_bridge, sample_opps
        )
        gaps = identify_coverage_gaps(completeness, sample_accounts)
        result = estimate_enrichment_pipeline(gaps)
        if len(result) > 0:
            assert "estimated_pipeline_uplift" in result.columns
            assert (result["estimated_pipeline_uplift"] >= 0).all()

    def test_empty_gaps(self):
        empty = pd.DataFrame()
        result = estimate_enrichment_pipeline(empty)
        assert len(result) == 0
