# Fraud Detection System — Project Brief for Claude Code

## Purpose

This is a portfolio project for **Liam Havers**, a Data Scientist (2+ years, MSc Data Science) currently strong in NLP/topic modelling/stats but light on **productionised, tabular ML**. Goal: build a credit-card fraud detection system that goes from raw data → trained model → served API → basic monitoring, to demonstrate end-to-end ML engineering skill for tech/finance Data Scientist job applications.

**Audience for the finished repo**: recruiters and hiring managers at tech/finance companies (e.g. fintechs, banks, Bloomberg-type firms) skimming GitHub. It needs to read as clearly production-minded, not just a notebook exercise.

**Success criteria**: a working local (or free-tier deployed) `/predict` API backed by a properly evaluated, imbalance-aware model, with clean repo structure, a strong README, Docker support, and basic monitoring/logging — completable in a focused few weeks, not months.

---

## Tech stack

- **Language**: Python 3.11+
- **Modelling**: XGBoost or LightGBM (pick one, justify choice in README)
- **Imbalance handling**: `imbalanced-learn` (SMOTE) and/or class weighting — implement both, compare, keep the better one
- **API**: FastAPI
- **Containerisation**: Docker
- **Data validation**: Pydantic (comes with FastAPI) for request schemas
- **Experiment tracking**: simple — log runs to a local `experiments/` folder as JSON/CSV; no need for MLflow unless it's trivial to add later
- **Testing**: pytest for at least the API endpoint and core feature transform functions
- **Monitoring**: lightweight custom logging + a basic drift check (see Phase 4)
- **Dataset**: Kaggle Credit Card Fraud Detection dataset (`mlg-ulb/creditcardfraud`) — anonymised PCA features, highly imbalanced (~0.17% fraud). Stretch goal: swap in or add IEEE-CIS fraud dataset for richer feature engineering if time allows.

---

## Repo structure (target)

```
fraud-detection-system/
├── CLAUDE.md
├── README.md
├── data/
│   ├── raw/                  # gitignored, download instructions in README
│   └── processed/            # gitignored
├── notebooks/
│   └── 01_eda.ipynb          # exploration only — no modelling logic lives here long-term
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── load.py           # download/load dataset
│   │   └── preprocess.py     # cleaning, feature engineering, train/test split
│   ├── models/
│   │   ├── train.py          # training pipeline (imbalance handling, model fit)
│   │   ├── evaluate.py       # PR-AUC, cost-sensitive threshold selection, plots
│   │   └── predict.py        # inference wrapper used by the API
│   ├── monitoring/
│   │   └── drift.py          # feature distribution drift check
│   └── config.py             # paths, thresholds, model hyperparams as one source of truth
├── api/
│   ├── main.py                # FastAPI app, /predict, /health endpoints
│   └── schemas.py             # Pydantic request/response models
├── models/
│   └── (saved model artifacts, gitignored — document how to regenerate)
├── tests/
│   ├── test_api.py
│   └── test_preprocess.py
├── experiments/                # logged run metrics (json/csv), not code
├── Dockerfile
├── docker-compose.yml          # optional — nice if adding monitoring later
├── requirements.txt
├── .gitignore
└── .github/
    └── workflows/
        └── ci.yml              # basic lint + test on push (stretch goal, do last)
```

---

## Build phases

Work through these roughly in order. Each phase should end in something runnable/demoable — don't let phases blur together.

### Phase 1 — Data & EDA
- Download the Kaggle Credit Card Fraud dataset, document the download step in README (Kaggle requires auth — don't commit the data).
- EDA notebook: class imbalance, feature distributions, correlation with target, no leakage checks.
- Time-aware train/test split (the dataset has a `Time` column — split so training data precedes test data, don't shuffle randomly, since this mirrors a real production scenario and is worth calling out in the README as a deliberate choice).

### Phase 2 — Modelling
- Baseline: logistic regression with class weighting, to have a reference point.
- Main model: XGBoost or LightGBM.
- Handle imbalance two ways (SMOTE on training data, and class-weighting) — compare results, pick a winner, document why in README/evaluate.py docstring.
- Hyperparameter tuning: keep it simple — a small grid/random search is fine, don't over-engineer this part.

### Phase 3 — Evaluation (this is where DS judgement shows — don't skip)
- Report PR-AUC as the primary metric (not accuracy — with ~0.17% positive class, accuracy is meaningless and a reviewer checking this file will notice if it's used as headline metric).
- Precision-recall curve, confusion matrix at a few candidate thresholds.
- **Cost-sensitive threshold selection**: define illustrative costs (e.g. false negative = £X average fraud loss, false positive = £Y customer friction/investigation cost), pick an operating threshold that minimises expected cost, and explain the trade-off in plain English in the README. This one paragraph is the single highest-signal piece of the whole project for interviews — it demonstrates business-aware modelling, not just metric-chasing.

### Phase 4 — Productionisation
- `predict.py`: clean inference wrapper — load model artifact, preprocess input, return prediction + probability + which threshold decision was applied.
- FastAPI `main.py`:
  - `POST /predict` — accepts a transaction's features, returns fraud probability + flagged boolean.
  - `GET /health` — basic liveness check.
  - Pydantic schema validation on input (reject malformed requests cleanly).
- Dockerfile: multi-stage if easy, single-stage is fine otherwise. Should `docker build` + `docker run` cleanly with no manual steps beyond that.
- Basic logging: every prediction request logged (timestamp, input summary, output) to a local file or stdout — this is what lets you claim "monitoring" honestly.
- `drift.py`: a simple feature-distribution comparison (e.g. KS-test or PSI) between a reference window and a simulated "new" window of data. Doesn't need to be sophisticated — just needs to exist and be explained, since **structured monitoring awareness** is a meaningful differentiator for finance/tech DS roles even at basic implementation depth.

### Phase 5 — Polish & packaging (this is what actually gets you the interview)
- **README.md** is the most important file in the repo. Structure it as:
  1. One-paragraph problem statement and why it matters (frame in business terms: cost of fraud, cost of false positives)
  2. Architecture diagram (simple — can be ASCII or a simple image, doesn't need to be fancy)
  3. Key modelling decisions and trade-offs (imbalance handling choice, threshold selection reasoning)
  4. Results (PR-AUC, chosen threshold, expected cost reduction vs a naive baseline)
  5. How to run it (data download → train → serve → test, exact commands)
  6. What you'd do next with more time (shows self-awareness/seniority — e.g. "add real-time feature store," "A/B test threshold in shadow mode")
- Add a couple of tests in `tests/` — doesn't need full coverage, just enough to show testing discipline (API returns 200 and expected schema, preprocessing function handles edge cases).
- **Stretch, only if time allows**: deploy on Render/Railway free tier and link a live demo URL in the README — this materially raises the "did they actually ship it" signal versus code-only.

---

## Constraints and conventions

- Keep `config.py` as the single source of truth for paths, chosen threshold, model hyperparameters — no magic numbers scattered through the codebase; interviewers who open the repo will notice this kind of hygiene.
- No data or model artifacts committed to git — `.gitignore` them, document regeneration steps.
- Prefer clarity over cleverness throughout — this repo will be read by humans deciding whether to interview a candidate, not just executed.
- Type hints on function signatures in `src/` — matches the standard expected in professional tech/finance codebases.
- Commit in logical, well-messaged chunks (not one giant commit) — commit history is sometimes checked by reviewers as evidence of real, incremental work rather than a single dump.

## What "done" looks like

- `docker build && docker run` starts the API with no manual steps beyond what's documented.
- `curl` or the FastAPI `/docs` Swagger UI can hit `/predict` with a sample transaction and get a sensible response.
- README tells the full story end-to-end without needing to read the code.
- PR-AUC and the cost-sensitive threshold reasoning are clearly reported and explained.

## Out of scope (don't scope-creep into these)

- Real-time streaming ingestion (Kafka etc.) — mention as a "next steps" line in README instead, don't build it.
- Full MLOps stack (MLflow, Airflow, Kubernetes) — too much for a portfolio project timeline, and reviewers will value a clean, complete simple system over a half-finished complex one.
- Frontend/UI — the API + Swagger docs is the interface; no need for a separate frontend.
