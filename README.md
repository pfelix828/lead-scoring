# Lead Scoring & Buying Group Identification

A propensity model and buying group framework for B2B SaaS marketing, built on synthetic data that mirrors real GTM workflows.

## The Problem

In B2B SaaS, marketing teams target thousands of accounts but only a fraction will close. The challenge is twofold:

1. **Which accounts are most likely to buy?** — Propensity scoring ranks accounts by conversion likelihood so marketing can allocate resources efficiently.
2. **What's preventing accounts from closing?** — Buying group analysis identifies *who's missing* at each account (roles, seniority, functions) and recommends specific enrichment actions.

This project builds both layers and combines them into a targeting framework that maps directly to how GTM teams operate.

## Key Results

### Propensity Model (Logistic Regression)

| Metric | Value |
|--------|-------|
| Test AUC | 0.608 |
| Precision @ Top 10% | 54.5% (vs. 36% baseline) |
| Top-Decile Lift | 1.4x |

The model uses 49 features across three layers — firmographic/technographic signals, contact composition metrics, and buying group characteristics. Top drivers include VP+ engagement on deals, complementary tech stack, and buying group completeness.

### Buying Group Analysis

| Completeness Tier | Win Rate | Accounts |
|-------------------|----------|----------|
| Complete (75-100) | 51% | 146 |
| High (50-75) | 38% | 176 |
| Medium (25-50) | 27% | 170 |
| Low (0-25) | 24% | 87 |

Accounts with complete buying groups win at **2x the rate** of those with low completeness. The gap analysis identified 464 accounts with specific coverage gaps (missing roles, seniority, or function diversity) and estimated **$286M in addressable pipeline** from enrichment.

## Project Structure

```
lead-scoring/
├── src/
│   ├── generate_data.py       # Synthetic CRM data (4 tables, ~13K rows)
│   ├── features.py            # Feature engineering (contact + account level)
│   ├── model.py               # Model training, evaluation, business metrics
│   └── buying_groups.py       # Completeness scoring & gap analysis
├── notebooks/
│   ├── 01_data_exploration    # EDA, signal hunting, initial hypotheses
│   ├── 02_feature_engineering # Feature derivation walkthrough + correlation analysis
│   ├── 03_modeling            # Train, evaluate, compare LR vs RF, business metrics
│   └── 04_buying_groups       # Completeness scoring, gap analysis, enrichment targets
├── tests/                     # 56 pytest tests
└── app/                       # Streamlit dashboard (TBD)
```

## How to Run

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate synthetic data
python src/generate_data.py

# Run tests
pytest tests/ -v

# Run notebooks (in order)
jupyter notebook notebooks/
```

## Methodology

### Synthetic Data
Four tables mirror a real CRM stack: **accounts** (1,500 companies with firmographic and technographic data), **contacts** (10,504 people with job titles, functions, seniority), **opportunities** (800 deals), and **contact_opportunity** (1,944 role assignments linking people to deals).

Win probability is influenced by realistic signals: VP+ contacts on the deal, technical + business function mix, complementary tech stack, buying group size, segment, and industry.

### Feature Engineering
Features are built at three levels, from weakest to strongest signal:
- **Firmographic/Technographic** — segment, employee count, revenue, tech stack flags, existing customer status
- **Contact Composition** — VP+ count, function diversity, senior density, technical + business mix
- **Buying Group** — deal contact count, role coverage, completeness score (0-100)

### Model Selection
Logistic regression was chosen over random forest despite marginal AUC improvement (0.608 vs 0.652) because:
- Coefficients translate directly to scoring rules ("VP+ engagement adds X points")
- Mirrors what production marketing stacks (Marketo, HubSpot) actually use
- Easier to explain to non-technical stakeholders
- Regularization (L2, C=0.01) handles feature redundancy

### Buying Group Framework
Each account's buying group is scored on four dimensions (25 points each):
- **Role coverage** — Champion, Decision Maker, Evaluator, Influencer
- **Seniority mix** — VP+ and Director/Manager level contacts
- **Function diversity** — contacts across 2+ departments
- **Technical + Business** — both builders and buyers at the table

Gap accounts are paired with propensity scores to create a 2×2 targeting framework that prioritizes enrichment where it will have the highest ROI.

## What I'd Do Differently in Production

- **Title normalization** — Real job titles are messy ("Sr. Dir. of Eng" vs "Senior Director, Engineering"). Would use fuzzy matching or an LLM to standardize before feature extraction.
- **Temporal validation** — Train on earlier time periods, test on later ones. Time-based splits are more realistic than random splits for forecasting use cases.
- **Live scoring pipeline** — Integrate with Salesforce/HubSpot via scheduled batch scoring or event-driven updates when new contacts are added.
- **Model monitoring** — Track AUC and calibration over time. Retrain when performance degrades (quarterly at minimum, or when the product/market changes).
- **A/B test the model** — Before full rollout, randomly assign accounts to model-scored vs. control groups and measure incremental lift in pipeline and bookings.
