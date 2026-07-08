# Credit Card Fraud Detection System

End-to-end fraud detection: raw transaction data → an imbalance-aware trained model → a served prediction API → basic production monitoring — plus a second, richer dataset for feature-engineering depth and a statistically rigorous offline A/B test.

> **Status: scaffolding complete, modelling in progress.** The repo structure, API skeleton, and tests are in place; the trained models, evaluation results, and A/B test writeup below will be filled in as each phase lands. See [What's implemented so far](#whats-implemented-so-far).

## Why this matters

Card issuers lose money two ways: missed fraud (direct loss, chargebacks, reputational damage) and false alarms (blocked legitimate transactions, customer churn, manual review cost). A fraud model isn't judged on accuracy — with fraud typically well under 1% of transactions, a model that predicts "not fraud" every time is 99%+ "accurate" and useless. The real job is finding an operating point that minimises *expected cost*, not maximising a vanity metric. That trade-off, made explicit, is the core deliverable of this project.

## Why two datasets

Most fraud-detection portfolio projects stop at the Kaggle Credit Card Fraud dataset — it's clean, fast to work with, and consequently used in hundreds of near-identical repos. It's kept here as the dataset behind the **served model**, because its small anonymised feature set is realistic for a low-latency `/predict` endpoint. Alongside it, this project also uses **IEEE-CIS Fraud Detection** — raw, non-anonymised transaction and identity fields — as a **modelling-depth showcase**: real categorical encoding, joins, and missing-data strategy, rather than fitting a model to precomputed PCA components. IEEE-CIS isn't wired into the live API (see [Architecture](#architecture) for why); it's evaluated and written up on its own.

## Why a shadow-mode A/B test

I wanted hands-on practice with the *statistics* behind A/B testing — confidence intervals, hypothesis testing, effect size — which I hadn't built before. A live, traffic-splitting production experiment isn't achievable in a portfolio project with no real users, so this project instead replays historical held-out data through two competing modelling policies (SMOTE-resampling vs. class-weighting) and statistically compares their outcomes offline. **I'm explicit that this is a simulation, not production A/B-testing experience**: it has no real traffic split, no protection against novelty or seasonality effects, and none of the online-experimentation infrastructure a live test would need. It's deliberate practice of the underlying statistical reasoning, done honestly rather than dressed up as something it isn't. See [Results](#results) for the write-up once Phase 3 lands.

## Architecture

```
                          ┌──────────────────┐
     Kaggle dataset   →   │  src/data/        │  load, clean, time-aware split
     (served model)       │  preprocess.py    │
                          └────────┬──────────┘
                                   │
     IEEE-CIS dataset  →  ┌────────▼──────────┐
     (showcase only,      │  src/data/         │  join, categorical encoding,
      not served)         │  preprocess_ieee.py│  missing-data strategy
                          └────────┬──────────┘
                                   │
                          ┌────────▼──────────┐
                          │  src/models/       │  SMOTE vs class-weighting,
                          │  train.py          │  XGBoost fit (both datasets)
                          └────────┬──────────┘
                                   │
                    ┌──────────────┼───────────────────┐
                    │              │                    │
          ┌─────────▼────────┐   ┌▼───────────────────┐│
          │  src/models/      │   │  src/models/        ││
          │  evaluate.py      │   │  ab_test.py          ││
          │  PR-AUC, cost-    │   │  shadow-mode         ││
          │  sensitive        │   │  statistical compare ││
          │  threshold        │   │  (bootstrap CI)      ││
          └─────────┬────────┘   └─────────────────────┘│
                    │  model.pkl (Kaggle only)            │
          ┌─────────▼────────┐                            │
          │  src/models/      │                            │
          │  predict.py       │                            │
          └─────────┬────────┘                            │
                    │                                      │
          ┌─────────▼────────┐        ┌──────────────────┐│
Client →  │  api/main.py       │  ────▶ │  src/monitoring/  ││
POST      │  FastAPI (Docker)  │  logs  │  drift.py (PSI)   ││
/predict  └───────────────────┘        └──────────────────┘│
                                                             │
IEEE-CIS results reported in README/notebook only ──────────┘
```

## Tech stack

| Concern | Choice |
|---|---|
| Modelling | XGBoost (chosen for native imbalanced-data handling via `scale_pos_weight`, strong tabular performance, and fast inference) |
| Imbalance handling | SMOTE (`imbalanced-learn`) and class-weighting, compared empirically via the shadow-mode A/B test — winner documented in `src/models/evaluate.py` and `src/models/ab_test.py` |
| Datasets | Kaggle Credit Card Fraud (served model) + IEEE-CIS Fraud Detection (feature-engineering showcase, not served) |
| Statistical testing | `scipy`/`numpy` — bootstrap confidence intervals and significance testing for the A/B comparison |
| API | FastAPI + Pydantic request validation |
| Containerisation | Docker |
| Experiment tracking | Flat JSON/CSV run logs in `experiments/` |
| Testing | pytest |
| Monitoring | Structured prediction logging + PSI-based feature drift check |

## Key modelling decisions

- **Time-aware train/test split.** Both datasets' time columns (`Time` for Kaggle, `TransactionDT` for IEEE-CIS) are used to split chronologically rather than shuffle randomly — training data precedes test data. This mirrors production, where a model only ever sees past transactions at training time; a random split would leak future distribution information into training and overstate performance.
- **PR-AUC over accuracy/ROC-AUC.** With ~0.17% positive class in the Kaggle data, accuracy is meaningless and ROC-AUC can look deceptively good due to the large true-negative volume. PR-AUC reported as the headline metric for both datasets.
- **Cost-sensitive threshold selection.** Rather than defaulting to a 0.5 probability cutoff, an operating threshold is chosen to minimise *expected cost* using illustrative unit costs (false negative ≈ average fraud loss, false positive ≈ customer friction / manual investigation cost). *Results and the full trade-off writeup land here once Phase 3 is complete.*
- **Shadow-mode A/B test instead of an eyeballed comparison.** SMOTE vs. class-weighting is decided via a formal offline statistical comparison (bootstrapped confidence interval on the expected-cost difference between the two trained policies, replayed over held-out data) rather than just picking whichever number is bigger. See [Why a shadow-mode A/B test](#why-a-shadow-mode-ab-test) for what this does and doesn't demonstrate.
- **IEEE-CIS kept separate from the served API.** Its raw schema is 400+ columns across two joined files — an unwieldy Pydantic request schema that would hurt the "Swagger docs as the interface" goal more than it would help. It's evaluated and written up as its own modelling exercise instead.

## Results

*To be filled in after training and evaluation (Phase 2–3):*

**Kaggle (served model)**
- PR-AUC: `TBD`
- Chosen decision threshold: `TBD`
- Expected cost reduction vs. naive baseline: `TBD`
- Precision / recall at chosen threshold: `TBD`

**IEEE-CIS (feature-engineering showcase)**
- PR-AUC: `TBD`
- Key engineered features and why they helped: `TBD`

**Shadow-mode A/B test (SMOTE vs. class-weighting)**
- Expected cost per policy, bootstrapped 95% CI on the difference: `TBD`
- Statistically distinguishable from noise? `TBD`
- Limitations of this simulation vs. a real online experiment: see [Why a shadow-mode A/B test](#why-a-shadow-mode-ab-test)

## What's implemented so far

- [x] Repo structure, `config.py` as single source of truth for paths/thresholds/hyperparameters
- [x] FastAPI skeleton (`/health` working, `/predict` schema-validated, inference pending trained model)
- [x] Test scaffolding (`pytest`, API + preprocessing tests)
- [x] Dockerfile / docker-compose
- [x] CI workflow (lint + test on push)
- [ ] Kaggle EDA notebook
- [ ] IEEE-CIS EDA notebook (categorical/missingness focus)
- [ ] Data loading, cleaning, time-aware split (both datasets)
- [ ] Baseline logistic regression + XGBoost training, SMOTE vs. class-weighting comparison
- [ ] IEEE-CIS feature engineering and training
- [ ] Evaluation: PR-AUC, confusion matrices, cost-sensitive threshold (both datasets)
- [x] Bootstrap confidence interval logic (`ab_test.py`) — statistical core in place, awaiting real trained-model costs to compare
- [ ] Shadow-mode A/B test full writeup (results + limitations paragraph)
- [ ] Wired-up `/predict` inference (Kaggle model)
- [ ] Drift monitoring (PSI)

## How to run it

### 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Get the data

This project uses two datasets, neither of which is committed to this repo.

**Kaggle Credit Card Fraud Detection** (`mlg-ulb/creditcardfraud`) — the served model:

1. Create a Kaggle account and API token (`kaggle.json`) — see [Kaggle API docs](https://www.kaggle.com/docs/api).
2. Place it at `~/.kaggle/kaggle.json`.
3. Download and place the CSV at `data/raw/creditcard/creditcard.csv`, e.g.:

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/creditcard --unzip
```

**IEEE-CIS Fraud Detection** (`ieee-fraud-detection`) — the feature-engineering showcase:

```bash
kaggle competitions download -c ieee-fraud-detection -p data/raw/ieee_cis --unzip
```

(Requires accepting the competition rules on Kaggle first.) This places `train_transaction.csv` and `train_identity.csv` in `data/raw/ieee_cis/`.

### 3. Train

```bash
python -m src.models.train
```

### 4. Run the shadow-mode A/B test

```bash
python -m src.models.ab_test
```

### 5. Serve

```bash
uvicorn api.main:app --reload
```

Then open `http://localhost:8000/docs` for interactive Swagger docs, or:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_transaction.json
```

### 6. Test

```bash
pytest tests/ -v
```

### Docker

```bash
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
```

## What I'd do next with more time

- Real-time feature store instead of point-in-time PCA features, to support richer engineered features (velocity, merchant-category aggregates).
- Productionise the A/B test: real traffic splitting, guardrail metrics, sequential-testing controls — the current version is deliberately an offline statistical exercise, not this.
- MLflow for experiment tracking once run volume outgrows flat-file logging.
- Streaming ingestion (Kafka) for near-real-time scoring instead of request/response batch scoring.
- Serve the IEEE-CIS model behind its own endpoint if there's a clean way to do it without an unwieldy request schema (e.g. a feature-store lookup by transaction ID instead of a raw 400-field payload).

## Repo structure

```
fraud-detection-system/
├── data/
│   ├── raw/
│   │   ├── creditcard/     # gitignored — Kaggle Credit Card Fraud CSV
│   │   └── ieee_cis/       # gitignored — IEEE-CIS transaction + identity CSVs
│   └── processed/          # gitignored
├── notebooks/
│   ├── 01_eda_creditcard.ipynb
│   └── 02_eda_ieee_cis.ipynb
├── src/
│   ├── data/                 # load, preprocess (Kaggle), preprocess_ieee (IEEE-CIS)
│   ├── models/                # train, evaluate, ab_test, predict
│   ├── monitoring/            # drift.py
│   └── config.py             # paths, thresholds, hyperparameters
├── api/                # FastAPI app + Pydantic schemas (Kaggle model only)
├── models/              # gitignored model artifacts
├── tests/
├── experiments/          # logged run metrics (json/csv)
├── Dockerfile
└── docker-compose.yml
```
