# Credit Card Fraud Detection System

End-to-end fraud detection: raw transaction data → an imbalance-aware trained model → a served prediction API → basic production monitoring — plus a second, richer dataset for feature-engineering depth and a statistically rigorous offline A/B test.

Built as a portfolio project demonstrating end-to-end, production-minded ML engineering for tech/finance Data Scientist roles.

## Problem Statement

> The real job isn't maximising accuracy — it's finding the operating point that minimises the *combined expected cost* of missed fraud and false alarms.

Card issuers lose money two ways: missed fraud (direct loss, chargebacks, reputational damage) and false alarms (blocked legitimate transactions, customer churn, manual review cost). With fraud typically well under 1% of transactions, a model that predicts "not fraud" every time is 99%+ "accurate" and useless — accuracy as a headline metric would be actively misleading here. That cost trade-off, made explicit, is the core deliverable of this project.

## Phase 0 Decisions (locked)

These decisions were made deliberately up front, to keep the project focused rather than open-ended.

- **Two datasets, two different jobs**: Most fraud-detection portfolio projects stop at the Kaggle Credit Card Fraud dataset — it's clean, fast to work with, and consequently used in hundreds of near-identical repos. It's kept here as the dataset behind the **served model**, because its small anonymised feature set is realistic for a low-latency `/predict` endpoint. Alongside it, this project also uses **IEEE-CIS Fraud Detection** — raw, non-anonymised transaction and identity fields — as a **modelling-depth showcase**: real categorical encoding, joins, and missing-data strategy, rather than fitting a model to precomputed PCA components. IEEE-CIS isn't wired into the live API (see [Architecture](#architecture) for why); it's evaluated and written up on its own.
- **Time-aware train/test split**: Both datasets' time columns (`Time` for Kaggle, `TransactionDT` for IEEE-CIS) are used to split chronologically rather than shuffle randomly — training data precedes test data. This mirrors production, where a model only ever sees past transactions at training time; a random split would leak future distribution information into training and overstate performance.
- **PR-AUC over accuracy/ROC-AUC**: With ~0.17% positive class in the Kaggle data, accuracy is meaningless and ROC-AUC can look deceptively good due to the large true-negative volume. PR-AUC is reported as the headline metric for both datasets.
- **Cost-sensitive threshold selection over a default 0.5 cutoff**: An operating threshold is chosen to minimise *expected cost* using illustrative unit costs (false negative ≈ average fraud loss, false positive ≈ customer friction / manual investigation cost), rather than defaulting to a 0.5 probability cutoff. *Results and the full trade-off writeup land here once Phase 3 is complete.*
- **Shadow-mode A/B test, not a live experiment**: I wanted hands-on practice with the *statistics* behind A/B testing — confidence intervals, hypothesis testing, effect size — which I hadn't built before. A live, traffic-splitting production experiment isn't achievable in a portfolio project with no real users, so this project instead replays historical held-out data through two competing modelling policies (SMOTE-resampling vs. class-weighting) and statistically compares their outcomes offline (bootstrapped confidence interval on the expected-cost difference, rather than just picking whichever number is bigger). **I'm explicit that this is a simulation, not production A/B-testing experience**: it has no real traffic split, no protection against novelty or seasonality effects, and none of the online-experimentation infrastructure a live test would need. It's deliberate practice of the underlying statistical reasoning, done honestly rather than dressed up as something it isn't.
- **IEEE-CIS kept out of the served API**: Its raw schema is 400+ columns across two joined files — an unwieldy Pydantic request schema that would hurt the "Swagger docs as the interface" goal more than it would help. It's evaluated and written up as its own modelling exercise instead.

## Goals

- Build an end-to-end pipeline from raw transaction data to a served, monitored prediction API
- Handle severe class imbalance two ways (SMOTE vs. class-weighting) and pick a winner with statistical evidence, not intuition
- Select an operating threshold via explicit cost-sensitive reasoning, not a default cutoff
- Add a second, richer dataset to demonstrate real feature-engineering practice beyond anonymised PCA components
- Practice the statistics behind A/B testing through an honest, clearly-scoped offline shadow-mode simulation
- Ship basic production monitoring (structured logging + drift detection) so "monitoring" is a real claim, not just a README line
- Document every decision and trade-off as clearly as the code itself

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

## Repo Structure

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
├── tests/                # test_preprocess, test_preprocess_ieee, test_ab_test, test_api
├── experiments/          # logged run metrics (json/csv)
├── Dockerfile
└── docker-compose.yml
```

## Project Plan

### Phase 0 — Scaffolding
- [x] Repo structure, `config.py` as single source of truth for paths/thresholds/hyperparameters
- [x] FastAPI skeleton (`/health` working, `/predict` schema-validated, inference pending trained model)
- [x] Test scaffolding (`pytest`, API + preprocessing tests)
- [x] Dockerfile / docker-compose
- [x] CI workflow (lint + test on push)

### Phase 1 — Data & EDA (complete)
- [x] Kaggle EDA notebook (class imbalance, feature distributions, correlation with target, leakage checks) — 284,807 rows, 1,081 exact duplicates dropped by `clean()`. A weak baseline using *only* `Time`+`Amount` scores ROC-AUC 0.58, confirming the real fraud signal lives in the PCA components, not superficial fields. The leakage check surfaced something worth being honest about rather than hiding: 12,446 rows across the full dataset share an identical feature fingerprint (V1–V28 + Amount) with another row under a different `Time` — but **every one of them is a legitimate transaction, zero are fraud**. Given the anonymised features make the root cause unconfirmable and it can't inflate the metric that matters here, this is documented as a known data-quality caveat rather than papered over with a guessed fix.
- [x] IEEE-CIS EDA notebook (categorical cardinality, missing-data patterns, engineered features) — 590,540 transactions, 3.5% fraud (~20x less imbalanced than Kaggle), only 24.4% with a matching identity record. Fraud rate differs sharply by that alone (7.8% with an identity match vs. 2.1% without), which is why `has_identity` is engineered as an explicit feature. 174/394 columns are >50% null; missingness is left as native `NaN` for XGBoost (no imputation) with explicit `_is_missing` flags added for the most-null columns, since for this dataset absence often reflects *how* a transaction was made, not noise.
- [x] Data loading, cleaning, time-aware train/test split (both datasets)
- [x] IEEE-CIS feature engineering (`preprocess_ieee.py`) — join, categorical encoding, card-level aggregations (`card1_frequency`, `card1_mean_amount`, `time_since_last_txn_same_card`). Encoders and stats are fit on the training split only and applied to test — fitting on the whole joined dataset before splitting would leak exactly the kind of cross-split information the Kaggle notebook's finding was a reminder to watch for.

### Phase 2 — Modelling
- [ ] Baseline logistic regression + XGBoost training on Kaggle, SMOTE vs. class-weighting comparison
- [ ] IEEE-CIS XGBoost training on the engineered feature set

### Phase 3 — Evaluation & Shadow-Mode A/B Testing
- [ ] PR-AUC, precision-recall curve, confusion matrix at candidate thresholds, cost-sensitive threshold selection (both datasets) — results to report here: PR-AUC, chosen decision threshold, expected cost reduction vs. a naive baseline, precision/recall at the chosen threshold (Kaggle); PR-AUC and key engineered features and why they helped (IEEE-CIS)
- [x] Bootstrap confidence interval logic (`ab_test.py`) — statistical core in place, awaiting real trained-model costs to compare
- [ ] Shadow-mode A/B test full writeup (SMOTE vs. class-weighting) — expected cost per policy, bootstrapped 95% CI on the difference, whether it's statistically distinguishable from noise, and the limitations of this simulation vs. a real online experiment (see [Phase 0 Decisions](#phase-0-decisions-locked))

### Phase 4 — Productionisation
- [ ] Wire up `/predict` inference with the trained Kaggle model
- [ ] `drift.py` — PSI-based feature drift check between a reference window and a simulated "new" window

### Phase 5 — Polish & Packaging
- [ ] Full results write-up in this README (PR-AUC, chosen threshold, expected cost reduction, both datasets)
- [ ] Stretch: deploy on Render/Railway free tier and link a live demo URL

## Tech Stack

- **Language**: Python 3.11+
- **Modelling**: XGBoost (chosen for native imbalanced-data handling via `scale_pos_weight`, strong tabular performance, and fast inference)
- **Imbalance handling**: SMOTE (`imbalanced-learn`) and class-weighting, compared empirically via the shadow-mode A/B test — winner documented in `src/models/evaluate.py` and `src/models/ab_test.py`
- **Datasets**: Kaggle Credit Card Fraud (served model) + IEEE-CIS Fraud Detection (feature-engineering showcase, not served)
- **Statistical testing**: `scipy`/`numpy` — bootstrap confidence intervals and significance testing for the A/B comparison
- **API**: FastAPI + Pydantic request validation
- **Containerisation**: Docker
- **Experiment tracking**: Flat JSON/CSV run logs in `experiments/`
- **Testing**: pytest
- **Monitoring**: Structured prediction logging + PSI-based feature drift check

## Key Design Principles

1. **No leakage, verified not assumed** — every split respects the time boundary, and every fitted encoder/statistic is fit on training data only. Checking this rather than assuming it is what caught the Kaggle fingerprint-duplicate finding above.
2. **PR-AUC over accuracy** — the headline metric must not be gameable by class imbalance.
3. **Cost-sensitive decisions over default thresholds** — the operating point is chosen deliberately, not defaulted to 0.5.
4. **Honest about limitations** — the shadow-mode A/B test is explicitly framed as statistical practice, not production experimentation; data-quality quirks are documented, not hidden.

## How to Run It

### 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Get the Data

This project uses two datasets, neither of which is committed to this repo.

**Kaggle Credit Card Fraud Detection** (`mlg-ulb/creditcardfraud`) — the served model:

1. Create a Kaggle account and API token (`kaggle.json`) — see [Kaggle API docs](https://www.kaggle.com/docs/api).
2. Place it at `~/.kaggle/kaggle.json`.
3. Download and place the CSV at `data/raw/creditcard/creditcard.csv`, e.g.:

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/creditcard --unzip
```

**IEEE-CIS Fraud Detection** (`ieee-fraud-detection`) — the feature-engineering showcase:

1. Accept the competition rules at [kaggle.com/c/ieee-fraud-detection/rules](https://www.kaggle.com/c/ieee-fraud-detection/rules) first — the API returns a 403 until you do.
2. Download and unzip (the competition API doesn't support `--unzip` the way dataset downloads do):

```bash
kaggle competitions download -c ieee-fraud-detection -p data/raw/ieee_cis
python -c "import zipfile; zipfile.ZipFile('data/raw/ieee_cis/ieee-fraud-detection.zip').extractall('data/raw/ieee_cis')"
rm data/raw/ieee_cis/ieee-fraud-detection.zip
```

This produces `train_transaction.csv`, `train_identity.csv`, plus unlabeled `test_*`/`sample_submission.csv` files from the Kaggle competition itself (leaderboard submission files with no fraud labels — this project does its own time-aware split of the labeled training data instead, so those can be deleted):

```bash
rm data/raw/ieee_cis/test_transaction.csv data/raw/ieee_cis/test_identity.csv data/raw/ieee_cis/sample_submission.csv
```

### 3. Train

```bash
python -m src.models.train
```

### 4. Run the Shadow-Mode A/B Test

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

## Next Steps

- Real-time feature store instead of point-in-time PCA features, to support richer engineered features (velocity, merchant-category aggregates).
- Productionise the A/B test: real traffic splitting, guardrail metrics, sequential-testing controls — the current version is deliberately an offline statistical exercise, not this.
- MLflow for experiment tracking once run volume outgrows flat-file logging.
- Streaming ingestion (Kafka) for near-real-time scoring instead of request/response batch scoring.
- Serve the IEEE-CIS model behind its own endpoint if there's a clean way to do it without an unwieldy request schema (e.g. a feature-store lookup by transaction ID instead of a raw 400-field payload).

## Status

🚧 In progress — Phase 0 (scaffolding) and Phase 1 (data & EDA) are complete for both datasets; Phase 2 (modelling) is next. Trained models, evaluation results, and the A/B test writeup will land as Phases 2–4 complete. See the checkboxes under [Project Plan](#project-plan) for exact status per phase.
