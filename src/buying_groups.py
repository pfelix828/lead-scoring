"""
Buying group formation and gap analysis.

Goes beyond individual lead scoring to answer:
- What does a complete buying group look like at an account?
- Which accounts have incomplete groups?
- Where should marketing focus enrichment efforts?
"""

import numpy as np
import pandas as pd

TECHNICAL_FUNCTIONS = {"Engineering", "IT", "Product"}
BUSINESS_FUNCTIONS = {"Marketing", "Sales", "Finance", "Executive"}
KEY_ROLES = ["Champion", "Decision Maker", "Evaluator", "Influencer"]


def score_buying_group_completeness(
    accounts: pd.DataFrame,
    contacts: pd.DataFrame,
    contact_opportunity: pd.DataFrame,
    opportunities: pd.DataFrame,
) -> pd.DataFrame:
    """
    Score each account's buying group completeness on a 0-100 scale.

    Dimensions (25 points each):
    - Role coverage: distinct deal roles present out of 4 key roles
    - Seniority mix: has both VP+ AND Director/Manager level contacts
    - Function diversity: contacts across 2+ distinct functions
    - Technical + Business: at least one technical AND one business function contact
    """
    # Get contacts involved in deals for each account
    deal_contacts = (
        contact_opportunity
        .merge(contacts[["contact_id", "seniority", "job_function"]], on="contact_id")
        .merge(opportunities[["opportunity_id", "account_id"]], on="opportunity_id")
    )

    if len(deal_contacts) == 0:
        return accounts[["account_id"]].assign(
            completeness_score=0,
            role_coverage_score=0,
            seniority_mix_score=0,
            function_diversity_score=0,
            tech_business_score=0,
            roles_present=[],
            roles_missing=[],
            has_vp_plus=False,
            has_mid_level=False,
            function_count=0,
            has_technical=False,
            has_business=False,
        )

    # Aggregate per account
    acct_groups = deal_contacts.groupby("account_id").agg(
        roles=("role", set),
        seniorities=("seniority", set),
        functions=("job_function", set),
        contact_count=("contact_id", "nunique"),
    ).reset_index()

    # Role coverage (0-25)
    acct_groups["roles_present"] = acct_groups["roles"].apply(
        lambda r: sorted(set(r) & set(KEY_ROLES))
    )
    acct_groups["roles_missing"] = acct_groups["roles"].apply(
        lambda r: sorted(set(KEY_ROLES) - set(r))
    )
    acct_groups["role_coverage_score"] = acct_groups["roles"].apply(
        lambda r: (len(set(r) & set(KEY_ROLES)) / len(KEY_ROLES)) * 25
    )

    # Seniority mix (0-25)
    acct_groups["has_vp_plus"] = acct_groups["seniorities"].apply(
        lambda s: bool(s & {"C-Suite", "VP"})
    )
    acct_groups["has_mid_level"] = acct_groups["seniorities"].apply(
        lambda s: bool(s & {"Director", "Manager"})
    )
    acct_groups["seniority_mix_score"] = (
        acct_groups["has_vp_plus"].astype(int) * 12.5
        + acct_groups["has_mid_level"].astype(int) * 12.5
    )

    # Function diversity (0-25)
    acct_groups["function_count"] = acct_groups["functions"].apply(len)
    acct_groups["function_diversity_score"] = np.minimum(
        acct_groups["function_count"] / 3 * 25, 25
    )

    # Technical + Business (0-25)
    acct_groups["has_technical"] = acct_groups["functions"].apply(
        lambda f: bool(f & TECHNICAL_FUNCTIONS)
    )
    acct_groups["has_business"] = acct_groups["functions"].apply(
        lambda f: bool(f & BUSINESS_FUNCTIONS)
    )
    acct_groups["tech_business_score"] = (
        (acct_groups["has_technical"] & acct_groups["has_business"]).astype(int) * 25
    )

    # Total completeness
    acct_groups["completeness_score"] = (
        acct_groups["role_coverage_score"]
        + acct_groups["seniority_mix_score"]
        + acct_groups["tech_business_score"]
        + acct_groups["function_diversity_score"]
    )

    # Merge back to all accounts (accounts without deals get 0)
    result = accounts[["account_id"]].merge(
        acct_groups.drop(columns=["roles", "seniorities", "functions"]),
        on="account_id",
        how="left",
    )

    # Fill accounts with no deals
    result["completeness_score"] = result["completeness_score"].fillna(0)
    for col in ["role_coverage_score", "seniority_mix_score",
                "function_diversity_score", "tech_business_score"]:
        result[col] = result[col].fillna(0)
    result["roles_present"] = result["roles_present"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    result["roles_missing"] = result["roles_missing"].apply(
        lambda x: x if isinstance(x, list) else KEY_ROLES.copy()
    )
    for col in ["has_vp_plus", "has_mid_level", "has_technical", "has_business"]:
        result[col] = result[col].fillna(False)
    result["function_count"] = result["function_count"].fillna(0).astype(int)
    result["contact_count"] = result["contact_count"].fillna(0).astype(int)

    return result


def identify_coverage_gaps(
    completeness_df: pd.DataFrame,
    accounts: pd.DataFrame,
    min_completeness: float = 0,
    max_completeness: float = 75,
) -> pd.DataFrame:
    """
    Identify accounts with incomplete buying groups and what they're missing.

    Returns accounts in the specified completeness range with their gaps
    and recommended enrichment actions.
    """
    gaps = completeness_df[
        (completeness_df["completeness_score"] >= min_completeness)
        & (completeness_df["completeness_score"] <= max_completeness)
        & (completeness_df["contact_count"] > 0)  # Has at least some deal activity
    ].copy()

    # Merge account details
    gaps = gaps.merge(
        accounts[["account_id", "company_name", "segment", "industry", "annual_revenue"]],
        on="account_id",
        how="left",
    )

    # Generate enrichment recommendations
    recommendations = []
    for _, row in gaps.iterrows():
        recs = []
        if row["roles_missing"]:
            recs.append(f"Missing roles: {', '.join(row['roles_missing'])}")
        if not row["has_vp_plus"]:
            recs.append("Need VP+ contact")
        if not row["has_technical"]:
            recs.append("Need technical function contact (Engineering/IT/Product)")
        if not row["has_business"]:
            recs.append("Need business function contact (Marketing/Sales/Finance)")
        recommendations.append("; ".join(recs) if recs else "No gaps")

    gaps["enrichment_recommendation"] = recommendations

    return gaps.sort_values("completeness_score", ascending=False)


def completeness_vs_win_rate(
    completeness_df: pd.DataFrame,
    opportunities: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze the relationship between buying group completeness and deal outcomes.

    Groups accounts into completeness tiers and computes win rates.
    """
    closed = opportunities[opportunities["stage"].isin(["Closed Won", "Closed Lost"])].copy()

    # Best outcome per account
    acct_outcome = closed.groupby("account_id").agg(
        is_won=("is_won", "max"),
        total_amount=("amount", "sum"),
        deal_count=("opportunity_id", "count"),
    ).reset_index()

    # Merge with completeness
    merged = completeness_df.merge(acct_outcome, on="account_id", how="inner")

    # Create tiers
    merged["completeness_tier"] = pd.cut(
        merged["completeness_score"],
        bins=[-1, 25, 50, 75, 100],
        labels=["Low (0-25)", "Medium (25-50)", "High (50-75)", "Complete (75-100)"],
    )

    summary = merged.groupby("completeness_tier", observed=False).agg(
        account_count=("account_id", "count"),
        win_rate=("is_won", "mean"),
        avg_deal_value=("total_amount", "mean"),
        total_pipeline=("total_amount", "sum"),
    ).reset_index()

    return summary


def estimate_enrichment_pipeline(
    gaps_df: pd.DataFrame,
    win_rate_uplift: float = 0.10,
) -> pd.DataFrame:
    """
    Estimate the pipeline value of enriching gap accounts.

    Assumes that filling coverage gaps would increase win rate by win_rate_uplift.
    """
    if len(gaps_df) == 0:
        return gaps_df

    result = gaps_df.copy()
    result["estimated_deal_value"] = result["annual_revenue"] * 0.02  # ~2% of revenue as deal size
    result["estimated_pipeline_uplift"] = result["estimated_deal_value"] * win_rate_uplift

    return result
