"""
Feature engineering for lead scoring and account propensity modeling.

Two levels of features:
- Contact-level: for individual lead scoring (which contacts to prioritize)
- Account-level: for account propensity and buying group analysis
"""

import numpy as np
import pandas as pd

# Tools that signal higher propensity
HIGH_SIGNAL_TOOLS = {"Jira", "Salesforce", "Slack", "Snowflake", "AWS"}
TOP_TOOLS = ["Salesforce", "Slack", "Jira", "AWS", "Azure", "GCP", "HubSpot",
             "Snowflake", "Databricks", "Tableau"]

TECHNICAL_FUNCTIONS = {"Engineering", "IT", "Product"}
BUSINESS_FUNCTIONS = {"Marketing", "Sales", "Finance", "Executive"}

SENIORITY_RANK = {
    "Individual Contributor": 1,
    "Manager": 2,
    "Director": 3,
    "VP": 4,
    "C-Suite": 5,
}


def build_contact_features(contacts: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    """
    Build contact-level feature matrix.

    Each row is a contact with engineered features from both the contact's
    own attributes and their account's firmographic/technographic data.
    """
    df = contacts.copy()

    # --- Contact-level features ---
    df["seniority_rank"] = df["seniority"].map(SENIORITY_RANK).fillna(0).astype(int)
    df["is_vp_plus"] = df["seniority"].isin(["C-Suite", "VP"]).astype(int)
    df["is_director_plus"] = df["seniority"].isin(["C-Suite", "VP", "Director"]).astype(int)
    df["is_technical"] = df["job_function"].isin(TECHNICAL_FUNCTIONS).astype(int)
    df["is_business"] = df["job_function"].isin(BUSINESS_FUNCTIONS).astype(int)

    # One-hot encode job function
    function_dummies = pd.get_dummies(df["job_function"], prefix="func")
    df = pd.concat([df, function_dummies], axis=1)

    # --- Merge account-level context ---
    acct_features = _build_account_firmographic(accounts)
    df = df.merge(acct_features, on="account_id", how="left")

    return df


def build_account_features(
    accounts: pd.DataFrame,
    contacts: pd.DataFrame,
    opportunities: pd.DataFrame,
    contact_opportunity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build account-level feature matrix for propensity scoring.

    Combines firmographic data, technographic signals, contact composition,
    and buying group characteristics.
    """
    df = _build_account_firmographic(accounts)

    # --- Contact composition features ---
    contact_agg = _build_contact_composition(contacts)
    df = df.merge(contact_agg, on="account_id", how="left")

    # --- Buying group features from deal history ---
    bg_features = _build_deal_contact_features(opportunities, contact_opportunity, contacts)
    df = df.merge(bg_features, on="account_id", how="left")

    # Fill NaN for accounts with no opportunities
    fill_cols = [c for c in df.columns if c not in ["account_id"]]
    df[fill_cols] = df[fill_cols].fillna(0)

    return df


def _build_account_firmographic(accounts: pd.DataFrame) -> pd.DataFrame:
    """Firmographic and technographic features at the account level."""
    df = accounts[["account_id"]].copy()

    # Segment one-hot
    seg_dummies = pd.get_dummies(accounts["segment"], prefix="seg")
    df = pd.concat([df, seg_dummies], axis=1)

    # Industry one-hot
    ind_dummies = pd.get_dummies(accounts["industry"], prefix="ind")
    df = pd.concat([df, ind_dummies], axis=1)

    # Size features (log-transformed)
    df["employee_count_log"] = np.log1p(accounts["employee_count"])
    df["annual_revenue_log"] = np.log1p(accounts["annual_revenue"])

    # Existing customer
    df["has_existing_product"] = accounts["has_existing_product"].astype(int)
    df["arr_log"] = np.log1p(accounts["arr"])

    # Technographic features
    tech_stacks = accounts["tech_stack"].apply(
        lambda ts: set(t.strip() for t in str(ts).split(","))
    )
    df["tech_stack_count"] = tech_stacks.apply(len)
    df["high_signal_tool_count"] = tech_stacks.apply(
        lambda tools: len(tools & HIGH_SIGNAL_TOOLS)
    )

    # Individual tool flags for top tools
    for tool in TOP_TOOLS:
        df[f"has_{tool.lower()}"] = tech_stacks.apply(
            lambda tools, t=tool: int(t in tools)
        )

    # Region one-hot
    region_dummies = pd.get_dummies(accounts["region"], prefix="region")
    df = pd.concat([df, region_dummies], axis=1)

    return df


def _build_contact_composition(contacts: pd.DataFrame) -> pd.DataFrame:
    """Aggregate contact-level data to account-level composition metrics."""
    agg = contacts.groupby("account_id").agg(
        contact_count=("contact_id", "count"),
        vp_plus_count=("seniority", lambda s: (s.isin(["C-Suite", "VP"])).sum()),
        director_plus_count=("seniority", lambda s: (s.isin(["C-Suite", "VP", "Director"])).sum()),
        function_diversity=("job_function", "nunique"),
        has_technical=("job_function", lambda f: int(any(x in TECHNICAL_FUNCTIONS for x in f))),
        has_business=("job_function", lambda f: int(any(x in BUSINESS_FUNCTIONS for x in f))),
        max_seniority=("seniority", lambda s: max(SENIORITY_RANK.get(x, 0) for x in s)),
    ).reset_index()

    agg["has_technical_and_business"] = (
        (agg["has_technical"] == 1) & (agg["has_business"] == 1)
    ).astype(int)
    agg["senior_density"] = agg["vp_plus_count"] / agg["contact_count"]

    return agg


def _build_deal_contact_features(
    opportunities: pd.DataFrame,
    contact_opportunity: pd.DataFrame,
    contacts: pd.DataFrame,
) -> pd.DataFrame:
    """Build features from deal contact roles (buying group signals)."""
    if len(contact_opportunity) == 0 or len(opportunities) == 0:
        return pd.DataFrame(columns=["account_id"])

    # Enrich bridge table
    deal_contacts = (
        contact_opportunity
        .merge(contacts[["contact_id", "seniority", "job_function"]], on="contact_id")
        .merge(opportunities[["opportunity_id", "account_id", "is_won"]], on="opportunity_id")
    )

    # Aggregate to account level
    acct_deal = deal_contacts.groupby("account_id").agg(
        deal_contact_count=("contact_id", "nunique"),
        deal_role_diversity=("role", "nunique"),
        has_champion=("role", lambda r: int("Champion" in set(r))),
        has_decision_maker=("role", lambda r: int("Decision Maker" in set(r))),
        has_evaluator=("role", lambda r: int("Evaluator" in set(r))),
        has_influencer=("role", lambda r: int("Influencer" in set(r))),
        deal_vp_plus=("seniority", lambda s: int(any(x in ["C-Suite", "VP"] for x in s))),
        deal_function_diversity=("job_function", "nunique"),
    ).reset_index()

    # Buying group completeness (0-100 scale)
    key_roles = ["Champion", "Decision Maker", "Evaluator", "Influencer"]
    role_coverage = deal_contacts.groupby("account_id")["role"].apply(
        lambda r: len(set(r) & set(key_roles)) / len(key_roles)
    ).reset_index(name="role_coverage")

    acct_deal = acct_deal.merge(role_coverage, on="account_id", how="left")
    acct_deal["buying_group_completeness"] = (
        acct_deal["role_coverage"] * 25
        + acct_deal["deal_vp_plus"] * 25
        + (acct_deal["deal_function_diversity"] >= 2).astype(int) * 25
        + acct_deal.apply(
            lambda row: 25 if row.get("has_champion", 0) and row.get("has_decision_maker", 0) else 0,
            axis=1
        )
    )

    return acct_deal


def get_modeling_dataset(
    accounts: pd.DataFrame,
    contacts: pd.DataFrame,
    opportunities: pd.DataFrame,
    contact_opportunity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build the final modeling dataset: account-level features with deal outcome labels.

    Returns:
        X: Feature matrix (account-level)
        y: Binary target (1 = won, 0 = lost)
    """
    # Build account features
    account_features = build_account_features(
        accounts, contacts, opportunities, contact_opportunity
    )

    # Get target: for accounts with closed deals, did they win?
    closed = opportunities[opportunities["stage"].isin(["Closed Won", "Closed Lost"])]
    # Take the best outcome per account (if any deal won, account is positive)
    account_target = closed.groupby("account_id")["is_won"].max().reset_index()
    account_target.columns = ["account_id", "target"]

    # Merge
    dataset = account_features.merge(account_target, on="account_id", how="inner")

    # Separate features and target
    y = dataset["target"].astype(int)
    X = dataset.drop(columns=["account_id", "target"])

    return X, y
