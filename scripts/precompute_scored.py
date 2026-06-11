"""Precompute the dashboard's artifacts from the committed seed-42 dataset.

The deployed app loads these four small files instead of crunching 717K
contacts at boot, which keeps it inside Streamlit Cloud's guaranteed 690MB.
Re-run after any change to the data, features, or model:

    .venv/bin/python scripts/precompute_scored.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scored_dataset import build_scored_artifacts

DATA_DIR = PROJECT_ROOT / "data"


def main() -> None:
    data = {
        "accounts": pd.read_parquet(DATA_DIR / "accounts.parquet"),
        "contacts": pd.read_parquet(DATA_DIR / "contacts.parquet"),
        "opportunities": pd.read_parquet(DATA_DIR / "opportunities.parquet"),
        "contact_opp": pd.read_parquet(DATA_DIR / "contact_opportunity.parquet"),
    }

    scored, importance, eval_result, cv_auc = build_scored_artifacts(data)

    scored.to_parquet(DATA_DIR / "scored.parquet", compression="zstd", index=False)
    importance.to_parquet(DATA_DIR / "importance.parquet", compression="zstd", index=False)

    metrics = dict(eval_result["metrics"])
    lift = metrics.pop("lift_by_decile")
    lift.to_parquet(DATA_DIR / "lift_by_decile.parquet", compression="zstd", index=False)

    scalars = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
    non_scalars = {k: v for k, v in metrics.items() if not isinstance(v, (int, float))}
    if non_scalars:
        raise SystemExit(f"Unserialized metric keys: {list(non_scalars)} — handle these first.")
    scalars["cv_auc"] = float(cv_auc)
    (DATA_DIR / "model_eval.json").write_text(json.dumps(scalars, indent=1))

    print(f"scored: {len(scored):,} accounts | importance: {len(importance)} features")
    print(f"metrics: auc={scalars['auc']:.3f} cv_auc={scalars['cv_auc']:.3f}")
    for f in ["scored.parquet", "importance.parquet", "lift_by_decile.parquet", "model_eval.json"]:
        print(f"  {f}: {(DATA_DIR / f).stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
