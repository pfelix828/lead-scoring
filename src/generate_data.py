"""
Synthetic data generator for B2B SaaS lead scoring.

Generates four tables mirroring a real CRM + marketing automation stack:
- accounts: company firmographic and technographic data
- contacts: individual people at accounts
- opportunities: sales deals
- contact_opportunity: bridge table linking contacts to deals with roles

Signals are baked in at realistic strength to produce AUC ~0.75-0.80.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42

# --- Constants ---

INDUSTRIES = [
    "Technology", "Financial Services", "Healthcare", "Manufacturing",
    "Retail", "Media", "Professional Services", "Education"
]
INDUSTRY_WEIGHTS = [0.25, 0.20, 0.12, 0.12, 0.10, 0.08, 0.08, 0.05]

REGIONS = ["NA", "EMEA", "APAC", "LATAM"]
REGION_WEIGHTS = [0.50, 0.25, 0.15, 0.10]

TECH_TOOLS = [
    "Salesforce", "Slack", "Jira", "AWS", "Azure", "GCP", "HubSpot",
    "Marketo", "Snowflake", "Databricks", "Tableau", "Looker",
    "Zendesk", "Intercom", "Stripe", "Okta", "Workday", "SAP",
    "ServiceNow", "Confluence", "Asana", "Monday", "Figma",
    "GitHub", "Docker"
]

# Tools that signal higher propensity (complementary tech stack)
HIGH_SIGNAL_TOOLS = {"Jira", "Salesforce", "Slack", "Snowflake", "AWS"}

SENIORITIES = ["C-Suite", "VP", "Director", "Manager", "Individual Contributor"]
SENIORITY_WEIGHTS = [0.05, 0.10, 0.15, 0.25, 0.45]

JOB_FUNCTIONS = [
    "Engineering", "IT", "Marketing", "Sales", "Product",
    "Finance", "Operations", "Executive"
]
FUNCTION_WEIGHTS = [0.20, 0.12, 0.15, 0.12, 0.10, 0.10, 0.13, 0.08]

TECHNICAL_FUNCTIONS = {"Engineering", "IT", "Product"}
BUSINESS_FUNCTIONS = {"Marketing", "Sales", "Finance", "Executive"}

TITLES_BY_FUNCTION_SENIORITY = {
    ("Engineering", "C-Suite"): ["CTO", "Chief Technology Officer"],
    ("Engineering", "VP"): ["VP of Engineering", "VP of Software Development"],
    ("Engineering", "Director"): ["Director of Engineering", "Engineering Director"],
    ("Engineering", "Manager"): ["Engineering Manager", "Software Development Manager"],
    ("Engineering", "Individual Contributor"): ["Software Engineer", "Senior Software Engineer", "Staff Engineer"],
    ("IT", "C-Suite"): ["CIO", "Chief Information Officer"],
    ("IT", "VP"): ["VP of IT", "VP of Information Technology"],
    ("IT", "Director"): ["Director of IT", "IT Director"],
    ("IT", "Manager"): ["IT Manager", "Systems Manager"],
    ("IT", "Individual Contributor"): ["IT Specialist", "Systems Administrator", "IT Analyst"],
    ("Marketing", "C-Suite"): ["CMO", "Chief Marketing Officer"],
    ("Marketing", "VP"): ["VP of Marketing", "VP of Demand Generation"],
    ("Marketing", "Director"): ["Director of Marketing", "Director of Demand Gen"],
    ("Marketing", "Manager"): ["Marketing Manager", "Campaign Manager"],
    ("Marketing", "Individual Contributor"): ["Marketing Specialist", "Marketing Analyst", "Content Strategist"],
    ("Sales", "C-Suite"): ["CRO", "Chief Revenue Officer"],
    ("Sales", "VP"): ["VP of Sales", "VP of Business Development"],
    ("Sales", "Director"): ["Director of Sales", "Sales Director"],
    ("Sales", "Manager"): ["Sales Manager", "Account Manager"],
    ("Sales", "Individual Contributor"): ["Account Executive", "Sales Development Rep", "BDR"],
    ("Product", "C-Suite"): ["CPO", "Chief Product Officer"],
    ("Product", "VP"): ["VP of Product", "VP of Product Management"],
    ("Product", "Director"): ["Director of Product", "Product Director"],
    ("Product", "Manager"): ["Product Manager", "Senior Product Manager"],
    ("Product", "Individual Contributor"): ["Product Analyst", "Associate Product Manager"],
    ("Finance", "C-Suite"): ["CFO", "Chief Financial Officer"],
    ("Finance", "VP"): ["VP of Finance", "VP of FP&A"],
    ("Finance", "Director"): ["Director of Finance", "Finance Director"],
    ("Finance", "Manager"): ["Finance Manager", "FP&A Manager"],
    ("Finance", "Individual Contributor"): ["Financial Analyst", "Accountant"],
    ("Operations", "C-Suite"): ["COO", "Chief Operating Officer"],
    ("Operations", "VP"): ["VP of Operations", "VP of Business Operations"],
    ("Operations", "Director"): ["Director of Operations", "Operations Director"],
    ("Operations", "Manager"): ["Operations Manager", "Business Operations Manager"],
    ("Operations", "Individual Contributor"): ["Operations Analyst", "Operations Specialist"],
    ("Executive", "C-Suite"): ["CEO", "President", "Managing Director"],
    ("Executive", "VP"): ["EVP", "SVP", "General Manager"],
    ("Executive", "Director"): ["Director of Strategy", "Director of Business Development"],
    ("Executive", "Manager"): ["Strategy Manager", "Chief of Staff"],
    ("Executive", "Individual Contributor"): ["Business Analyst", "Strategy Analyst"],
}

LEAD_SOURCES = ["paid_search", "organic", "events", "outbound", "partner", "plg"]
LEAD_SOURCE_WEIGHTS = [0.20, 0.25, 0.15, 0.15, 0.10, 0.15]

DEAL_ROLES = ["Champion", "Decision Maker", "Influencer", "Evaluator", "Blocker"]

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Christopher", "Karen", "Charles", "Lisa", "Daniel", "Nancy",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
    "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle",
    "Kenneth", "Carol", "Kevin", "Amanda", "Brian", "Dorothy", "George", "Melissa",
    "Timothy", "Deborah", "Ronald", "Stephanie", "Edward", "Rebecca", "Jason", "Sharon",
    "Jeffrey", "Laura", "Ryan", "Cynthia", "Jacob", "Kathleen", "Gary", "Amy",
    "Nicholas", "Angela", "Eric", "Shirley", "Jonathan", "Anna", "Stephen", "Brenda",
    "Wei", "Priya", "Raj", "Aisha", "Carlos", "Yuki", "Jin", "Fatima",
    "Omar", "Mei", "Amit", "Sonia", "Luis", "Nadia", "Hassan", "Ling",
    "Diego", "Ananya", "Chen", "Olga", "Jamal", "Sakura", "Ravi", "Elena",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
    "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill",
    "Flores", "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell",
    "Mitchell", "Carter", "Roberts", "Patel", "Chen", "Kim", "Wang", "Singh",
    "Kumar", "Nakamura", "Tanaka", "Muller", "Schmidt", "Khan", "Ali", "Ahmed",
    "Santos", "Silva", "Costa", "Sato", "Park", "Choi", "Yang", "Huang",
    "Sharma", "Gupta", "Das", "Mehta", "Reyes", "Cruz", "Ramos", "Morales",
]

OPP_STAGES = ["Discovery", "Evaluation", "Proposal", "Negotiation", "Closed Won", "Closed Lost"]


def generate_accounts(n: int = 1500, rng: np.random.Generator = None) -> pd.DataFrame:
    """Generate synthetic account (company) data with firmographic and technographic fields."""
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    # Employee count: log-normal distribution
    employee_count = np.clip(
        rng.lognormal(mean=6.5, sigma=1.5, size=n).astype(int),
        20, 50000
    )

    # Segment derived from employee count
    segment = pd.cut(
        employee_count,
        bins=[0, 200, 2000, 100000],
        labels=["SMB", "Mid-Market", "Enterprise"]
    ).astype(str)

    # Revenue correlated with employee count (revenue per employee ~$100K-$300K with noise)
    rev_per_emp = rng.uniform(80_000, 350_000, size=n)
    annual_revenue = (employee_count * rev_per_emp).astype(int)

    # Tech stack: more tools for larger companies
    def pick_tech_stack(emp_count):
        base = 2 + int(np.log2(max(emp_count, 20)))
        n_tools = min(rng.poisson(base), len(TECH_TOOLS))
        n_tools = max(n_tools, 1)
        return ", ".join(rng.choice(TECH_TOOLS, size=n_tools, replace=False))

    tech_stacks = [pick_tech_stack(e) for e in employee_count]

    # Existing customers: ~18% overall, higher for Enterprise
    existing_probs = np.where(segment == "Enterprise", 0.30,
                     np.where(segment == "Mid-Market", 0.20, 0.12))
    has_existing = rng.random(n) < existing_probs

    # ARR for existing customers
    arr = np.where(
        has_existing,
        np.clip(rng.lognormal(mean=10, sigma=1.2, size=n), 5000, 500000),
        0
    ).astype(int)

    accounts = pd.DataFrame({
        "account_id": [f"ACC-{i:05d}" for i in range(1, n + 1)],
        "company_name": [f"Company {i}" for i in range(1, n + 1)],
        "industry": rng.choice(INDUSTRIES, size=n, p=INDUSTRY_WEIGHTS),
        "employee_count": employee_count,
        "segment": segment,
        "region": rng.choice(REGIONS, size=n, p=REGION_WEIGHTS),
        "annual_revenue": annual_revenue,
        "tech_stack": tech_stacks,
        "has_existing_product": has_existing,
        "arr": arr,
    })

    return accounts


def generate_contacts(accounts: pd.DataFrame, avg_per_account: float = 6.5,
                      rng: np.random.Generator = None) -> pd.DataFrame:
    """Generate synthetic contact data linked to accounts."""
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED + 1)

    contacts = []
    contact_id = 1

    for _, acct in accounts.iterrows():
        # More contacts for larger accounts
        segment_multiplier = {"SMB": 0.6, "Mid-Market": 1.0, "Enterprise": 1.8}
        n_contacts = max(2, int(rng.poisson(avg_per_account * segment_multiplier[acct["segment"]])))

        for _ in range(n_contacts):
            function = rng.choice(JOB_FUNCTIONS, p=FUNCTION_WEIGHTS)
            seniority = rng.choice(SENIORITIES, p=SENIORITY_WEIGHTS)

            title_key = (function, seniority)
            title = rng.choice(TITLES_BY_FUNCTION_SENIORITY[title_key])

            first = rng.choice(FIRST_NAMES)
            last = rng.choice(LAST_NAMES)
            domain = acct["company_name"].lower().replace(" ", "") + ".com"

            created = pd.Timestamp("2024-01-01") + pd.Timedelta(
                days=int(rng.integers(0, 790))
            )

            contacts.append({
                "contact_id": f"CON-{contact_id:06d}",
                "account_id": acct["account_id"],
                "first_name": first,
                "last_name": last,
                "email": f"{first.lower()}.{last.lower()}@{domain}",
                "job_title": title,
                "job_function": function,
                "seniority": seniority,
                "department": function,
                "lead_source": rng.choice(LEAD_SOURCES, p=LEAD_SOURCE_WEIGHTS),
                "created_at": created,
            })
            contact_id += 1

    return pd.DataFrame(contacts)


def generate_opportunities(accounts: pd.DataFrame, contacts: pd.DataFrame,
                           n_opps: int = 800,
                           rng: np.random.Generator = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate opportunities and the contact_opportunity bridge table.

    Win probability is influenced by:
    - Account segment (Mid-Market slightly higher than Enterprise/SMB)
    - Tech stack (high-signal tools increase win rate)
    - Contact seniority mix on the deal
    - Number of contacts involved (buying group completeness)
    - Whether technical AND business functions are represented
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED + 2)

    # Pick accounts that have opportunities (not all accounts have deals)
    acct_ids = accounts["account_id"].values
    opp_account_ids = rng.choice(acct_ids, size=n_opps, replace=True)

    opportunities = []
    bridge_rows = []

    for i, acct_id in enumerate(opp_account_ids):
        opp_id = f"OPP-{i + 1:06d}"
        acct = accounts[accounts["account_id"] == acct_id].iloc[0]
        acct_contacts = contacts[contacts["account_id"] == acct_id]

        if len(acct_contacts) == 0:
            continue

        # --- Assign contacts to this opportunity ---
        # Number of contacts on the deal: 1-6, weighted by account size
        max_contacts = min(len(acct_contacts), 6)
        n_deal_contacts = max(1, min(int(rng.poisson(2.5)), max_contacts))
        deal_contact_indices = rng.choice(len(acct_contacts), size=n_deal_contacts, replace=False)
        deal_contacts = acct_contacts.iloc[deal_contact_indices]

        # Assign roles
        available_roles = list(DEAL_ROLES)
        for _, contact in deal_contacts.iterrows():
            role = rng.choice(available_roles)
            bridge_rows.append({
                "contact_id": contact["contact_id"],
                "opportunity_id": opp_id,
                "role": role,
            })

        # --- Calculate win probability based on signals ---
        base_win_prob = 0.10

        # Signal 1: Account segment
        segment_bonus = {"SMB": -0.02, "Mid-Market": 0.04, "Enterprise": 0.00}
        win_prob = base_win_prob + segment_bonus.get(acct["segment"], 0)

        # Signal 2: Tech stack (high-signal tools)
        acct_tools = set(t.strip() for t in acct["tech_stack"].split(","))
        high_signal_count = len(acct_tools & HIGH_SIGNAL_TOOLS)
        win_prob += high_signal_count * 0.02

        # Signal 3: Seniority of contacts on the deal
        seniorities_on_deal = set(deal_contacts["seniority"].values)
        has_vp_plus = bool(seniorities_on_deal & {"C-Suite", "VP"})
        if has_vp_plus:
            win_prob += 0.10

        # Signal 4: Number of contacts (buying group size)
        if n_deal_contacts >= 3:
            win_prob += 0.07
        elif n_deal_contacts >= 2:
            win_prob += 0.02

        # Signal 5: Technical + Business function mix
        functions_on_deal = set(deal_contacts["job_function"].values)
        has_technical = bool(functions_on_deal & TECHNICAL_FUNCTIONS)
        has_business = bool(functions_on_deal & BUSINESS_FUNCTIONS)
        if has_technical and has_business:
            win_prob += 0.06

        # Signal 6: Industry
        industry_bonus = {
            "Technology": 0.03, "Financial Services": 0.02,
            "Healthcare": 0.00, "Manufacturing": -0.02,
            "Retail": -0.01, "Media": 0.01,
            "Professional Services": 0.01, "Education": -0.03
        }
        win_prob += industry_bonus.get(acct["industry"], 0)

        # Signal 7: Existing customer (upsell/cross-sell)
        if acct["has_existing_product"]:
            win_prob += 0.05

        # Clip and determine outcome
        win_prob = np.clip(win_prob, 0.03, 0.55)
        is_won = rng.random() < win_prob

        # Deal amount varies by segment
        amount_params = {
            "SMB": (8.5, 0.8),        # ~$5K-$20K
            "Mid-Market": (9.5, 0.9),  # ~$15K-$80K
            "Enterprise": (11.0, 1.0), # ~$50K-$500K
        }
        mean, sigma = amount_params.get(acct["segment"], (9.0, 1.0))
        amount = int(np.clip(rng.lognormal(mean, sigma), 2000, 600000))

        # Deal cycle days
        cycle_params = {"SMB": (30, 10), "Mid-Market": (55, 15), "Enterprise": (90, 25)}
        cycle_mean, cycle_std = cycle_params.get(acct["segment"], (55, 15))
        deal_cycle = max(7, int(rng.normal(cycle_mean, cycle_std)))

        created_at = pd.Timestamp("2024-03-01") + pd.Timedelta(
            days=int(rng.integers(0, 700))
        )
        closed_at = created_at + pd.Timedelta(days=deal_cycle) if is_won or rng.random() > 0.15 else pd.NaT

        # Stage
        if is_won:
            stage = "Closed Won"
        elif pd.notna(closed_at):
            stage = "Closed Lost"
        else:
            stage = rng.choice(["Discovery", "Evaluation", "Proposal", "Negotiation"])

        opportunities.append({
            "opportunity_id": opp_id,
            "account_id": acct_id,
            "primary_contact_id": deal_contacts.iloc[0]["contact_id"],
            "stage": stage,
            "amount": amount,
            "created_at": created_at,
            "closed_at": closed_at,
            "is_won": is_won,
            "deal_cycle_days": deal_cycle,
        })

    opp_df = pd.DataFrame(opportunities)
    bridge_df = pd.DataFrame(bridge_rows)

    return opp_df, bridge_df


def generate_all(output_dir: str = None) -> dict[str, pd.DataFrame]:
    """Generate all tables and optionally save to CSV."""
    rng = np.random.default_rng(RANDOM_SEED)

    print("Generating accounts...")
    accounts = generate_accounts(n=1500, rng=rng)

    print("Generating contacts...")
    contacts = generate_contacts(accounts, rng=np.random.default_rng(RANDOM_SEED + 1))

    print("Generating opportunities and contact roles...")
    opportunities, contact_opportunity = generate_opportunities(
        accounts, contacts, n_opps=800,
        rng=np.random.default_rng(RANDOM_SEED + 2)
    )

    tables = {
        "accounts": accounts,
        "contacts": contacts,
        "opportunities": opportunities,
        "contact_opportunity": contact_opportunity,
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            path = out / f"{name}.csv"
            df.to_csv(path, index=False)
            print(f"  Saved {name}: {len(df)} rows -> {path}")

    print(f"\nDone. Accounts: {len(accounts)}, Contacts: {len(contacts)}, "
          f"Opportunities: {len(opportunities)}, Contact roles: {len(contact_opportunity)}")

    return tables


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
    generate_all(output_dir=str(data_dir))
