# Lead Scoring Project — Claude Code Instructions

## What This Is
Portfolio project #2 — Lead scoring and buying group identification for B2B SaaS marketing. Mirrors work done at Atlassian on ICP and Buying Group identification.

## Stack
Python 3.12+ | Pandas | scikit-learn | Matplotlib | Streamlit | Jupyter

## Key Directories
- `src/` — Core Python modules (generate_data, features, model, buying_groups)
- `tests/` — pytest unit tests
- `notebooks/` — Jupyter notebooks (numbered, run in order)
- `data/` — Generated CSVs (gitignored, regenerable via `python src/generate_data.py`)
- `app/` — Streamlit dashboard

## Running
```bash
python src/generate_data.py    # generate synthetic data
pytest tests/ -v               # run tests
streamlit run app/streamlit_app.py  # launch dashboard
```

## Conventions
- Business framing first, technical implementation second
- Logistic regression is the primary model (interpretable, defensible)
- All notebooks should run top-to-bottom without external dependencies beyond data generation
