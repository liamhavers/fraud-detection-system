# Fraud Detection System — Project Brief for Claude Code

## Purpose

This is a portfolio project for **Liam Havers**, a Data Scientist (2+ years, MSc Data Science) currently strong in NLP/topic modelling/stats but light on **productionised, tabular ML**. Goal: build a credit-card fraud detection system that goes from raw data → trained model → served API → basic monitoring, to demonstrate end-to-end ML engineering skill for tech/finance Data Scientist job applications.

Two secondary learning goals shape the scope beyond a minimal single-dataset project:

1. **Real feature engineering depth.** A single anonymised PCA dataset (Kaggle) risks reading as generic — there are hundreds of near-identical portfolio repos built on it. Adding the **IEEE-CIS Fraud Detection** dataset alongside it gives a second, richer modelling exercise built on raw categorical/identity/merchant fields that require genuine feature engineering (encoding, joins, missing-data strategy), not just fitting a model to precomputed PCA components.
2. **Statistical rigour behind A/B testing.** Liam explicitly wants to build understanding of the *statistics* behind A/B testing (hypothesis testing, confidence intervals, effect size, power), which he currently lacks hands-on experience with. This project implements an **offline shadow-mode simulation** — replaying historical held-out data through two competing policies and statistically comparing outcomes — as a deliberate, scoped way to practice that statistical reasoning. See [Phase 3](#phase-3--evaluation-and-shadow-mode-ab-testing-this-is-where-ds-judgement-shows--dont-skip) and [Out of scope](#out-of-scope-dont-scope-creep-into-these) for what this is *not*.

**Audience for the finished repo**: recruiters and hiring managers at tech/finance companies (e.g. fintechs, banks, Bloomberg-type firms) skimming GitHub. It needs to read as clearly production-minded, not just a notebook exercise.

**Success criteria**: a working local (or free-tier deployed) `/predict` API backed by a properly evaluated, imbalance-aware model, with clean repo structure, a strong README, Docker support, and basic monitoring/logging. Scope has been deliberately expanded to two datasets plus a statistical A/B-testing exercise — timeline is not the binding constraint here, but each phase should still end in something runnable/demoable rather than sprawling into an open-ended research project.

---

## Tech stack

- **Language**: Python 3.11+
- **Modelling**: XGBoost (pick one, justify choice in README)
- **Imbalance handling**: `imbalanced-learn` (SMOTE) and/or class weighting — implement both, compare, keep the better one
- **API**: FastAPI
- **Containerisation**: Docker
- **Data validation**: Pydantic (comes with FastAPI) for request schemas
- **Experiment tracking**: simple — log runs to a local `experiments/` folder as JSON/CSV; no need for MLflow unless it's trivial to add later
- **Statistical testing**: `scipy`/`numpy` for bootstrap confidence intervals and hypothesis tests (shadow-mode A/B comparison — see Phase 3)
- **Testing**: pytest for at least the API endpoint and core feature transform functions
- **Monitoring**: lightweight custom logging + a basic drift check (see Phase 4)
- **Datasets** (two, serving different purposes — see [Purpose](#purpose)):
  - **Kaggle Credit Card Fraud Detection** (`mlg-ulb/creditcardfraud`) — anonymised PCA features, highly imbalanced (~0.17% fraud). This is the **served model**: small, clean, fast — realistic for a low-latency `/predict` endpoint.
  - **IEEE-CIS Fraud Detection** (`ieee-fraud-detection` Kaggle competition) — raw transaction + identity fields (400+ columns across two files, joined on `TransactionID`), extensive categorical encoding and missing-data handling required. This is a **modelling-depth showcase**: its own EDA, preprocessing, and evaluation, documented in the README, but **not wired into the live API** (a 400+-field Pydantic schema would hurt the Swagger-UI-as-interface goal more than it would help — see Phase 4).

---

## Repo structure (target)

```
fraud-detection-system/
├── CLAUDE.md
├── README.md
├── data/
│   ├── raw/
│   │   ├── creditcard/        # gitignored — Kaggle Credit Card Fraud CSV
│   │   └── ieee_cis/          # gitignored — IEEE-CIS transaction + identity CSVs
│   └── processed/             # gitignored
├── notebooks/
│   ├── 01_eda_creditcard.ipynb   # Kaggle dataset exploration only
│   └── 02_eda_ieee_cis.ipynb     # IEEE-CIS exploration — categorical/missingness focus
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── load.py               # download/load for both datasets
│   │   ├── preprocess.py         # Kaggle: cleaning, time-aware split
│   │   └── preprocess_ieee.py    # IEEE-CIS: join, categorical encoding, missing-data strategy, feature engineering
│   ├── models/
│   │   ├── train.py              # training pipeline (imbalance handling, model fit) — both datasets
│   │   ├── evaluate.py           # PR-AUC, cost-sensitive threshold selection, plots
│   │   ├── ab_test.py            # offline shadow-mode statistical comparison (bootstrap CI / hypothesis test)
│   │   └── predict.py            # inference wrapper used by the API (Kaggle model only)
│   ├── monitoring/
│   │   └── drift.py              # feature distribution drift check
│   └── config.py                 # paths, thresholds, model hyperparams as one source of truth
├── api/
│   ├── main.py                   # FastAPI app, /predict, /health endpoints (Kaggle model)
│   └── schemas.py                # Pydantic request/response models
├── models/
│   └── (saved model artifacts, gitignored — document how to regenerate)
├── tests/
│   ├── test_api.py
│   ├── test_preprocess.py
│   ├── test_preprocess_ieee.py
│   └── test_ab_test.py           # statistical comparison logic — sanity-check on synthetic data
├── experiments/                   # logged run metrics (json/csv), not code
├── Dockerfile
├── docker-compose.yml             # optional — nice if adding monitoring later
├── requirements.txt
├── .gitignore
└── .github/
    └── workflows/
        └── ci.yml                 # basic lint + test on push
```

---

## Build phases

Work through these roughly in order. Each phase should end in something runnable/demoable — don't let phases blur together.

### Phase 1 — Data & EDA
- **Kaggle dataset**: download the Credit Card Fraud dataset, document the download step in README (Kaggle requires auth — don't commit the data). EDA notebook: class imbalance, feature distributions, correlation with target, no leakage checks. Time-aware train/test split (the dataset has a `Time` column — split so training data precedes test data, don't shuffle randomly, since this mirrors a real production scenario and is worth calling out in the README as a deliberate choice).
- **IEEE-CIS dataset**: download `train_transaction.csv` + `train_identity.csv`, join on `TransactionID`. Separate EDA notebook focused on what this dataset actually demands: categorical cardinality, missing-data patterns (identity fields are sparsely populated — decide and document an explicit imputation/flagging strategy), and candidate engineered features (aggregations by card/merchant, time-since-last-transaction). Time-aware split using `TransactionDT`, same rationale as the Kaggle split.

### Phase 2 — Modelling
- **Kaggle**: baseline logistic regression with class weighting for a reference point, then XGBoost as the main model. Handle imbalance two ways (SMOTE on training data, and class-weighting) — this comparison feeds directly into the Phase 3 shadow-mode A/B test, so keep both trained variants around rather than discarding the loser immediately.
- **IEEE-CIS**: XGBoost on the engineered feature set from Phase 1. The bar here is *demonstrating real feature engineering practice*, not chasing competition-leaderboard performance — don't over-invest in squeezing out marginal score gains.
- Hyperparameter tuning: keep it simple — a small grid/random search is fine, don't over-engineer this part.

### Phase 3 — Evaluation and shadow-mode A/B testing (this is where DS judgement shows — don't skip)
- Report PR-AUC as the primary metric (not accuracy — with ~0.17% positive class, accuracy is meaningless and a reviewer checking this file will notice if it's used as headline metric).
- Precision-recall curve, confusion matrix at a few candidate thresholds.
- **Cost-sensitive threshold selection**: define illustrative costs (e.g. false negative = £X average fraud loss, false positive = £Y customer friction/investigation cost), pick an operating threshold that minimises expected cost, and explain the trade-off in plain English in the README. This one paragraph is one of the highest-signal pieces of the whole project for interviews — it demonstrates business-aware modelling, not just metric-chasing.
- **Shadow-mode A/B test (`src/models/ab_test.py`)**: instead of picking the SMOTE-vs-class-weighting winner by eyeballing metrics, treat it as a formal statistical comparison. Replay the held-out, time-ordered Kaggle test set through both trained policies, compute expected cost per policy, and compare using bootstrapped confidence intervals (and/or a paired significance test) on the cost difference. Report effect size and whether the difference is statistically distinguishable from noise, not just which number is bigger.
  - **This is explicitly an offline simulation, not a production experiment** — see [Out of scope](#out-of-scope-dont-scope-creep-into-these). The purpose is building statistical intuition (confidence intervals, hypothesis testing, sample-size/power considerations), and the README must say so plainly rather than implying live experimentation experience that wasn't actually exercised.

### Phase 4 — Productionisation
- `predict.py`: clean inference wrapper — load model artifact, preprocess input, return prediction + probability + which threshold decision was applied. **Kaggle model only** — the IEEE-CIS model is a modelling-depth showcase documented via notebook/README, not served live (see [Tech stack](#tech-stack) for why).
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
  3. Key modelling decisions and trade-offs (imbalance handling choice, threshold selection reasoning, why two datasets and what each is for)
  4. Results — both datasets: Kaggle (PR-AUC, chosen threshold, expected cost reduction vs a naive baseline) and IEEE-CIS (PR-AUC, key engineered features and why they helped)
  5. **Shadow-mode A/B test writeup**: the statistical comparison result, plus an explicit, honest paragraph on what this simulation does and doesn't demonstrate (no live traffic split, no production experiment infrastructure, no protection against novelty/seasonality effects a real online test would need to control for) — framed as deliberate practice of the underlying statistics, not a claim of production A/B-testing experience.
  6. How to run it (data download → train → serve → test, exact commands)
  7. What you'd do next with more time (shows self-awareness/seniority — e.g. "add real-time feature store," "productionise the A/B test with real traffic splitting and guardrail metrics")
- Add a couple of tests in `tests/` — doesn't need full coverage, just enough to show testing discipline (API returns 200 and expected schema, preprocessing function handles edge cases, A/B comparison logic sanity-checked on synthetic data with a known effect).
- **Stretch, only if time allows**: deploy on Render/Railway free tier and link a live demo URL in the README — this materially raises the "did they actually ship it" signal versus code-only.

---

## Constraints and conventions

- Keep `config.py` as the single source of truth for paths, chosen threshold, model hyperparameters — no magic numbers scattered through the codebase; interviewers who open the repo will notice this kind of hygiene.
- No data or model artifacts committed to git — `.gitignore` them, document regeneration steps.
- Prefer clarity over cleverness throughout — this repo will be read by humans deciding whether to interview a candidate, not just executed.
- Type hints on function signatures in `src/` — matches the standard expected in professional tech/finance codebases.
- Commit in logical, well-messaged chunks (not one giant commit) — commit history is sometimes checked by reviewers as evidence of real, incremental work rather than a single dump.
- Keep the two datasets' code paths clearly separated (`preprocess.py` vs `preprocess_ieee.py`, distinct notebooks) rather than forcing a single generic pipeline to handle both — they have genuinely different shapes and purposes, and a false-generic abstraction would obscure that rather than clarify it.

## What "done" looks like

- `docker build && docker run` starts the API with no manual steps beyond what's documented.
- `curl` or the FastAPI `/docs` Swagger UI can hit `/predict` with a sample transaction and get a sensible response.
- README tells the full story end-to-end without needing to read the code.
- PR-AUC and the cost-sensitive threshold reasoning are clearly reported and explained, for both datasets.
- The shadow-mode A/B test has a result, a stated confidence interval / significance result, and an honest limitations paragraph.

## Out of scope (don't scope-creep into these)

- **Live/production A/B testing infrastructure** — real traffic splitting, an online experimentation platform, sequential testing with peeking controls, guardrail-metric dashboards. The A/B testing work in this project is an offline shadow-mode simulation over historical data, done specifically to build statistical understanding — not a live experimentation system. Don't build toward making it look like one.
- **Competition-leaderboard performance on IEEE-CIS** — the goal is demonstrating real feature engineering practice, not maximising Kaggle competition rank; don't sink disproportionate time into marginal score gains here.
- Real-time streaming ingestion (Kafka etc.) — mention as a "next steps" line in README instead, don't build it.
- Full MLOps stack (MLflow, Airflow, Kubernetes) — too much for a portfolio project, and reviewers will value a clean, complete simple system over a half-finished complex one.
- Frontend/UI — the API + Swagger docs is the interface; no need for a separate frontend.
