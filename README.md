# Lead Scoring & Buying Group Identification

A propensity model and buying group framework for B2B SaaS marketing, built on synthetic data that mirrors real GTM workflows.

## The Problem

In B2B SaaS, marketing teams target tens of thousands of accounts but only a fraction will close. The challenge is twofold:

1. **Which accounts are most likely to buy?** — Propensity scoring ranks accounts by conversion likelihood so marketing can allocate resources efficiently.
2. **What's preventing accounts from closing?** — Buying group analysis identifies *who's missing* at each account (roles, seniority, functions) and recommends specific enrichment actions.

This project builds both layers and combines them into a targeting framework that maps directly to how GTM teams operate.

## Key Results

### Propensity Model (Logistic Regression)

| Metric | Value |
|--------|-------|
| Test AUC | 0.575 (95% CI: 0.56–0.59) |
| Precision @ Top 10% | 44.8% (95% CI: 0.41–0.48) |
| Top-Decile Lift | 1.3x |

All confidence intervals are bootstrap estimates: 1,000 resamples of the held-out test set, reported as the 2.5th and 97.5th percentiles of the metric across resamples.

Trained on ~36K accounts with closed deals (out of 100K total). The model uses pre-deal features only — firmographic/technographic signals and contact composition metrics. In-deal buying group features (from the contact-opportunity bridge) are excluded to avoid target leakage, since they are derived from the same opportunities whose outcome is the target. Top drivers include senior contact density, existing customer status, complementary tech stack, and segment.

AUC of ~0.58 on pre-deal features is realistic for production B2B propensity models — firmographic and contact signals are noisy in isolation. The value comes from concentrating wins in top deciles for more efficient resource allocation across 100K accounts. Random forest was tested as a benchmark (AUC 0.567) but showed no improvement over logistic regression, so LR was selected for its interpretable coefficients that translate directly to scoring rules in marketing stacks like Marketo and HubSpot.

#### Does the model earn its complexity?

A reported AUC only means something against a floor. The full 39-feature model was compared to four naive baselines on the same held-out test set, each isolating a progressively richer signal:

| Model | Features | Test AUC | 95% CI |
|-------|----------|----------|--------|
| Random | 0 | 0.494 | 0.480–0.507 |
| Senior density (rank only) | 1 | 0.548 | 0.535–0.562 |
| Segment only | 3 | 0.529 | 0.517–0.541 |
| Firmographic only | 30 | 0.562 | 0.549–0.575 |
| Full model | 39 | 0.575 | 0.561–0.589 |

A single feature, senior contact density, gets to AUC 0.548, about half the gap from random to the full model. Firmographic and technographic features alone reach 0.562. The contact-composition features lift AUC to 0.575, but the confidence intervals overlap (firmographic-only 0.549–0.575 vs full 0.561–0.589), so that lift is small and not cleanly distinguishable on this data. The honest read: most of the pre-deal propensity signal is firmographic, and the contact-composition features add marginal lift at the account level. The stronger signal from who is on the deal shows up in the buying group analysis below, where in-deal features are used descriptively rather than for prediction.

#### Calibration

AUC measures ranking; it says nothing about whether a predicted 40% actually means 40%. On the test set the predicted probabilities track observed win rates closely: across score deciles the largest gap between predicted and actual win rate is 2.4 percentage points, and the Brier score is 0.226 versus 0.230 for a no-skill model that always predicts the base rate. So the scores are weak at separating winners from losers but trustworthy as probabilities, which is what matters if marketing sets a threshold on them. The reliability plot is in notebook 03.

### Buying Group Analysis

| Completeness Tier | Win Rate | Accounts |
|-------------------|----------|----------|
| Complete (75-100) | 49% | 9,356 |
| High (50-75) | 39% | 10,467 |
| Medium (25-50) | 28% | 11,029 |
| Low (0-25) | 22% | 5,121 |

Accounts with complete buying groups win at **2.2x the rate** of those with low completeness. This is an association, not a causal estimate: complete buying groups tend to co-occur with larger, more engaged, later-stage accounts that were more likely to win regardless, so the raw 2.2x overstates the lift a team would get purely from filling gaps. Isolating the causal effect would need a controlled test (randomized enrichment) or, short of that, adjustment for account size, segment, and deal stage. The gap analysis identified 29,512 accounts with specific coverage gaps (missing roles, seniority, or function diversity). The 2×2 targeting framework (propensity × completeness) identifies 7,068 high-propensity accounts with incomplete buying groups as the highest-ROI enrichment targets — accounts the model predicts are likely to buy, but that need the right people at the table.

### Synthetic Data Limitations

The synthetic data generator explicitly encodes signals (VP+ contacts increase win probability, complementary tech stacks correlate with conversion, etc.) that the model then recovers. This means the results above are a **design validation** — confirming the model correctly recovers planted signals — not empirical discovery of real-world patterns. With real CRM data, signals would be noisier, confounders (sales rep quality, deal timing, competitive pressure) would exist, and temporal dynamics would matter. The model architecture and evaluation framework transfer directly to production; the specific coefficients and lift numbers do not.

The 100K-account scale mirrors realistic B2B SaaS GTM databases. At this volume, the operational focus shifts from manual account review to automated scoring pipelines, programmatic enrichment via data vendors (ZoomInfo, Apollo), and systematic gap alerts integrated into CRM workflows.

## Project Structure

```
lead-scoring/
├── src/
│   ├── generate_data.py       # Synthetic CRM data (4 tables, ~875K rows)
│   ├── features.py            # Feature engineering (contact + account level)
│   ├── model.py               # Model training, evaluation, business metrics
│   └── buying_groups.py       # Completeness scoring & gap analysis
├── notebooks/
│   ├── 01_data_exploration    # EDA, signal hunting, initial hypotheses
│   ├── 02_feature_engineering # Feature derivation walkthrough + correlation analysis
│   ├── 03_modeling            # Train, evaluate, compare LR vs RF, business metrics
│   └── 04_buying_groups       # Completeness scoring, gap analysis, enrichment targets
├── tests/                     # 61 pytest tests
└── app/                       # Streamlit dashboard
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
Four tables mirror a real CRM stack: **accounts** (100,000 companies with firmographic and technographic data), **contacts** (717,820 people with job titles, functions, seniority), **opportunities** (50,000 deals), and **contact_opportunity** (122,677 role assignments linking people to deals).

Win probability is influenced by realistic signals: VP+ contacts on the deal, technical + business function mix, complementary tech stack, buying group size, segment, and industry.

### Feature Engineering
Features are built at three levels, from weakest to strongest signal:
- **Firmographic/Technographic** — segment, employee count, revenue, tech stack flags, existing customer status
- **Contact Composition** — VP+ count, function diversity, senior density, technical + business mix
- **Buying Group** — deal contact count, role coverage, completeness score (0-100)

### Model Selection
Logistic regression was chosen over random forest — at 100K-account scale, LR slightly outperforms RF (AUC 0.575 vs 0.567):
- Coefficients translate directly to scoring rules ("senior density adds X points")
- Mirrors what production marketing stacks (Marketo, HubSpot) actually use
- Easier to explain to non-technical stakeholders
- Regularization (L2, C=0.01) handles feature redundancy
- No evidence of non-linear interactions that would justify a less interpretable model

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
