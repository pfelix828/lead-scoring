"""Export web-ready JSON for the Next.js app in web/.

Recomputes everything from the committed seed-42 parquet data through the
same code paths the analysis used (src/model.py, src/features.py), so every
number in the web app is reproduced, not transcribed. Slow part is the
baselines (four bootstrap-CI fits); ~1-2 minutes total.

Usage: .venv/bin/python scripts/export_web.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.buying_groups import score_buying_group_completeness  # noqa: E402
from src.features import get_modeling_dataset  # noqa: E402
from src.model import train_baselines  # noqa: E402

DATA = PROJECT_ROOT / "data"
OUT = PROJECT_ROOT / "web" / "src" / "data"

COMPLETENESS_TIERS = [
    (0, 25, "Low (0-25)"),
    (25, 50, "Medium (25-50)"),
    (50, 75, "High (50-75)"),
    (75, 101, "Complete (75-100)"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    scored = pd.read_parquet(DATA / "scored.parquet")
    importance = pd.read_parquet(DATA / "importance.parquet")
    lift = pd.read_parquet(DATA / "lift_by_decile.parquet")
    model_eval = json.loads((DATA / "model_eval.json").read_text())

    raw = {
        "accounts": pd.read_parquet(DATA / "accounts.parquet"),
        "contacts": pd.read_parquet(DATA / "contacts.parquet"),
        "opportunities": pd.read_parquet(DATA / "opportunities.parquet"),
        "contact_opp": pd.read_parquet(DATA / "contact_opportunity.parquet"),
    }

    # --- Baselines ladder: rerun the real code on the real split ---
    X, y = get_modeling_dataset(
        raw["accounts"], raw["contacts"], raw["opportunities"], raw["contact_opp"]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Computing baselines (bootstrap CIs, ~1 min)...")
    ladder = train_baselines(X_train, y_train, X_test, y_test)
    baselines = ladder.to_dict(orient="records")
    baselines.append({
        "model": "Full model (LR)",
        "n_features": int(X.shape[1]),
        "auc": model_eval["auc"],
        "auc_ci_lower": model_eval["auc_ci_lower"],
        "auc_ci_upper": model_eval["auc_ci_upper"],
    })
    print(ladder[["model", "auc"]].to_string(index=False))

    # --- Calibration on the held-out test set (same binning as plot_calibration) ---
    test = scored[scored.in_test_set]
    cal_df = pd.DataFrame({"y": test.target.values, "score": test.propensity_score.values})
    cal_df["bin"] = pd.qcut(cal_df["score"], 10, labels=False, duplicates="drop")
    cal_bins = (
        cal_df.groupby("bin")
        .agg(predicted=("score", "mean"), actual=("y", "mean"), count=("y", "count"))
        .reset_index(drop=True)
    )
    base_rate = float(cal_df["y"].mean())
    calibration = {
        "bins": cal_bins.round(4).to_dict(orient="records"),
        "max_gap": float((cal_bins.predicted - cal_bins.actual).abs().max()),
        "brier": float(brier_score_loss(cal_df["y"], cal_df["score"])),
        "brier_no_skill": float(brier_score_loss(cal_df["y"], np.full(len(cal_df), base_rate))),
        "test_n": int(len(cal_df)),
    }

    # --- Score distribution by outcome (test set, 25 bins) ---
    bins = np.linspace(0, 1, 26)
    hist = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = test[(test.propensity_score >= lo) & (test.propensity_score < hi)]
        hist.append({
            "bin": round(float((lo + hi) / 2), 3),
            "won": int((in_bin.target == 1).sum()),
            "lost": int((in_bin.target == 0).sum()),
        })

    # --- 2x2 framework (full scored population, matching the README definitions) ---
    median_prop = float(scored.propensity_score.median())
    with_contacts = scored[scored.contact_count > 0]
    quad = lambda hp, hc: with_contacts[  # noqa: E731
        (with_contacts.propensity_score >= median_prop if hp else with_contacts.propensity_score < median_prop)
        & (with_contacts.completeness_score >= 50 if hc else with_contacts.completeness_score < 50)
    ]
    quadrants = {
        "high_prop_low_complete": len(quad(True, False)),
        "high_prop_high_complete": len(quad(True, True)),
        "low_prop_low_complete": len(quad(False, False)),
        "low_prop_high_complete": len(quad(False, True)),
    }
    rng = np.random.default_rng(7)
    sample_idx = rng.choice(len(with_contacts), size=min(1500, len(with_contacts)), replace=False)
    scatter = [
        {
            "x": round(float(r.completeness_score), 1),
            "y": round(float(r.propensity_score), 4),
            "won": int(r.target),
        }
        for r in with_contacts.iloc[sample_idx].itertuples()
    ]

    # --- Buying-group tiers: same pd.cut bins as src/buying_groups.py
    #     (right-inclusive, full scored population) so the README table reproduces ---
    tier_labels = [label for _, _, label in COMPLETENESS_TIERS]
    tier_cut = pd.cut(scored.completeness_score, bins=[-1, 25, 50, 75, 100], labels=tier_labels)
    tiers = [
        {
            "tier": label,
            "win_rate": round(float(scored.target[tier_cut == label].mean()), 4),
            "n": int((tier_cut == label).sum()),
        }
        for label in tier_labels
    ]
    # Gap pool per identify_coverage_gaps: completeness <= 75 with contacts,
    # across the FULL account base (not just accounts with closed deals).
    completeness_full = score_buying_group_completeness(
        raw["accounts"], raw["contacts"], raw["contact_opp"], raw["opportunities"]
    )
    gap_pool_n = int(len(completeness_full[
        (completeness_full.completeness_score <= 75) & (completeness_full.contact_count > 0)
    ]))

    # --- Sub-scores by segment ---
    sub_cols = {
        "role_coverage_score": "Role coverage",
        "seniority_mix_score": "Seniority mix",
        "function_diversity_score": "Function diversity",
        "tech_business_score": "Technical + business",
    }
    sub_scores = []
    for seg, grp in with_contacts.groupby("segment"):
        entry = {"segment": seg}
        for col, label in sub_cols.items():
            entry[label] = round(float(grp[col].mean()), 2)
        sub_scores.append(entry)

    # --- Gap analysis (high propensity, incomplete group, has contacts) ---
    gaps = with_contacts[
        (with_contacts.propensity_score >= median_prop) & (with_contacts.completeness_score < 50)
    ]
    missing_counter: Counter[str] = Counter()
    for roles in gaps.roles_missing:
        items = list(roles) if not isinstance(roles, str) else [r.strip() for r in roles.split(",") if r.strip()]
        missing_counter.update(items)
    structural = {
        "No VP+ contact": int((~gaps.has_vp_plus.astype(bool)).sum()),
        "No technical contact": int((~gaps.has_technical.astype(bool)).sum()),
        "No business contact": int((~gaps.has_business.astype(bool)).sum()),
    }
    target_list = [
        {
            "company": r.company_name,
            "segment": r.segment,
            "industry": r.industry,
            "propensity": round(float(r.propensity_score), 3),
            "completeness": round(float(r.completeness_score), 0),
            "missing": (list(r.roles_missing) if not isinstance(r.roles_missing, str)
                        else [x.strip() for x in r.roles_missing.split(",") if x.strip()]),
            "has_vp": bool(r.has_vp_plus),
        }
        for r in gaps.sort_values("propensity_score", ascending=False).head(30).itertuples()
    ]
    gap_summary = {
        "gap_pool_n": gap_pool_n,
        "n_targets": int(len(gaps)),
        "avg_propensity": round(float(gaps.propensity_score.mean()), 4),
        "avg_completeness": round(float(gaps.completeness_score.mean()), 1),
        "median_propensity_threshold": round(median_prop, 4),
        "missing_roles": [{"role": k, "n": v} for k, v in missing_counter.most_common(10)],
        "structural": [{"gap": k, "n": v} for k, v in structural.items()],
    }

    # --- Assemble files ---
    payload = {
        "meta.json": {
            "accounts_total": int(len(raw["accounts"])),
            "scored_n": int(len(scored)),
            "test_n": int(test.shape[0]),
            "with_contacts_n": int(len(with_contacts)),
            "win_rate_overall": round(float(scored.target.mean()), 4),
        },
        "model.json": {
            "metrics": model_eval,
            "baselines": baselines,
            "calibration": calibration,
            "lift": lift.round(4).to_dict(orient="records"),
            "importance": importance.head(20).round(4).to_dict(orient="records"),
            "score_hist": hist,
        },
        "framework.json": {
            "median_propensity": round(median_prop, 4),
            "quadrants": quadrants,
            "scatter": scatter,
            "tiers": tiers,
            "sub_scores": sub_scores,
        },
        "targets.json": {"summary": gap_summary, "rows": target_list},
    }
    for name, obj in payload.items():
        path = OUT / name
        path.write_text(json.dumps(obj, indent=1, default=float))
        print(f"{name}: {path.stat().st_size / 1024:.0f} KB")

    print(f"\nparity vs README: quadrant = {quadrants['high_prop_low_complete']:,} (7,070) | "
          f"tiers = {[t['n'] for t in tiers]} (5,121/11,029/10,467/9,356) | "
          f"tier win rates = {[round(t['win_rate']*100) for t in tiers]} (22/28/39/49) | "
          f"gap pool = {gap_pool_n:,} (29,512) | "
          f"max calib gap = {calibration['max_gap']*100:.1f}pp (2.5)")


if __name__ == "__main__":
    main()
