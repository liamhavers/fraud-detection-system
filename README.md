# Credit Card Fraud Detection System

End-to-end fraud detection: raw transaction data → an imbalance-aware trained model → a served prediction API → basic production monitoring — plus a second, richer dataset for feature-engineering depth and a statistically rigorous offline A/B test.

> **Status: Phase 1 (data & EDA) complete, modelling in progress.** Both datasets are downloaded, cleaned, time-split, and explored in `notebooks/`; `src/data/preprocess_ieee.py`'s feature engineering is implemented and leakage-checked. Trained models, evaluation results, and the A/B test writeup below will be filled in as Phases 2–4 land. See [What's implemented so far](#whats-implemented-so-far).

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

## Phase 1 findings (EDA)

Full detail and charts live in `notebooks/01_eda_creditcard.ipynb` and `notebooks/02_eda_ieee_cis.ipynb`; headline findings:

**Kaggle** — 284,807 rows, 1,081 exact duplicates dropped by `clean()`. A weak baseline using *only* `Time`+`Amount` scores ROC-AUC 0.58, confirming the real fraud signal lives in the PCA components, not superficial fields. The leakage check surfaced something worth being honest about rather than hiding: 12,446 rows across the full dataset share an identical feature fingerprint (V1–V28 + Amount) with another row under a different `Time` — but **every one of them is a legitimate transaction, zero are fraud**. Given the anonymised features make the root cause unconfirmable and it can't inflate the metric that matters here, this is documented as a known data-quality caveat rather than papered over with a guessed fix.

**IEEE-CIS** — 590,540 transactions, 3.5% fraud (~20x less imbalanced than Kaggle), only 24.4% with a matching identity record. Fraud rate differs sharply by that alone (7.8% with an identity match vs. 2.1% without), which is why `has_identity` is engineered as an explicit feature. Feature encoders and card-level aggregations (`card1_frequency`, `card1_mean_amount`, `time_since_last_txn_same_card`) are fit on the training split only and applied to test — fitting on the whole joined dataset before splitting would leak exactly the kind of cross-split information the Kaggle notebook's finding was a reminder to watch for. 174/394 columns are >50% null; missingness is left as native `NaN` for XGBoost (no imputation) with explicit `_is_missing` flags added for the most-null columns, since for this dataset absence often reflects *how* a transaction was made, not noise.

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
- [x] Kaggle EDA notebook (class imbalance, distributions, correlation, leakage checks)
- [x] IEEE-CIS EDA notebook (cardinality, missingness, engineered features)
- [x] Data loading, cleaning, time-aware split (both datasets)
- [x] IEEE-CIS feature engineering (`preprocess_ieee.py`) — encoders fit on train only, applied to test
- [ ] Baseline logistic regression + XGBoost training, SMOTE vs. class-weighting comparison
- [ ] IEEE-CIS XGBoost training
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
├── tests/                # test_preprocess, test_preprocess_ieee, test_ab_test, test_api
├── experiments/          # logged run metrics (json/csv)
├── Dockerfile
└── docker-compose.yml
```
