# Credit Card Fraud Detection System

End-to-end fraud detection: raw transaction data → an imbalance-aware trained model → a served prediction API → basic production monitoring.

> **Status: scaffolding complete, modelling in progress.** The repo structure, API skeleton, and tests are in place; the trained model, evaluation results, and cost-sensitive threshold below will be filled in as each phase lands. See [What's implemented so far](#whats-implemented-so-far).

## Why this matters

Card issuers lose money two ways: missed fraud (direct loss, chargebacks, reputational damage) and false alarms (blocked legitimate transactions, customer churn, manual review cost). A fraud model isn't judged on accuracy — with fraud typically well under 1% of transactions, a model that predicts "not fraud" every time is 99%+ "accurate" and useless. The real job is finding an operating point that minimises *expected cost*, not maximising a vanity metric. That trade-off, made explicit, is the core deliverable of this project.

## Architecture

```
                    ┌──────────────────┐
  Kaggle dataset →  │  src/data/        │  load, clean, time-aware split
                    │  preprocess.py    │
                    └────────┬──────────┘
                             │
                    ┌────────▼──────────┐
                    │  src/models/       │  SMOTE vs class-weighting,
                    │  train.py          │  XGBoost fit
                    └────────┬──────────┘
                             │
                    ┌────────▼──────────┐
                    │  src/models/       │  PR-AUC, confusion matrix,
                    │  evaluate.py       │  cost-sensitive threshold
                    └────────┬──────────┘
                             │  model.pkl
                    ┌────────▼──────────┐
                    │  src/models/       │  inference wrapper
                    │  predict.py        │
                    └────────┬──────────┘
                             │
                    ┌────────▼──────────┐        ┌──────────────────┐
   Client request → │  api/main.py       │  ────▶ │  src/monitoring/  │
   POST /predict    │  FastAPI (Docker)  │  logs  │  drift.py (PSI)   │
                    └───────────────────┘        └──────────────────┘
```

## Tech stack

| Concern | Choice |
|---|---|
| Modelling | XGBoost (chosen for native imbalanced-data handling via `scale_pos_weight`, strong tabular performance, and fast inference — justification expanded once benchmarked against the LightGBM alternative) |
| Imbalance handling | SMOTE (`imbalanced-learn`) and class-weighting, compared empirically — winner documented in `src/models/evaluate.py` |
| API | FastAPI + Pydantic request validation |
| Containerisation | Docker |
| Experiment tracking | Flat JSON/CSV run logs in `experiments/` |
| Testing | pytest |
| Monitoring | Structured prediction logging + PSI-based feature drift check |

## Key modelling decisions

- **Time-aware train/test split.** The dataset's `Time` column is used to split chronologically rather than shuffle randomly — training data precedes test data. This mirrors production, where a model only ever sees past transactions at training time; a random split would leak future distribution information into training and overstate performance.
- **PR-AUC over accuracy/ROC-AUC.** With ~0.17% positive class, accuracy is meaningless and ROC-AUC can look deceptively good due to the large true-negative volume. PR-AUC reported as the headline metric.
- **Cost-sensitive threshold selection.** Rather than defaulting to a 0.5 probability cutoff, an operating threshold is chosen to minimise *expected cost* using illustrative unit costs (false negative ≈ average fraud loss, false positive ≈ customer friction / manual investigation cost). *Results and the full trade-off writeup land here once Phase 3 is complete.*

## Results

*To be filled in after training and evaluation (Phase 2–3):*

- PR-AUC: `TBD`
- Chosen decision threshold: `TBD`
- Expected cost reduction vs. naive baseline: `TBD`
- Precision / recall at chosen threshold: `TBD`

## What's implemented so far

- [x] Repo structure, `config.py` as single source of truth for paths/thresholds/hyperparameters
- [x] FastAPI skeleton (`/health` working, `/predict` schema-validated, inference pending trained model)
- [x] Test scaffolding (`pytest`, API + preprocessing tests)
- [x] Dockerfile / docker-compose
- [x] CI workflow (lint + test on push)
- [ ] EDA notebook
- [ ] Data loading, cleaning, time-aware split
- [ ] Baseline logistic regression + XGBoost training, SMOTE vs. class-weighting comparison
- [ ] Evaluation: PR-AUC, confusion matrices, cost-sensitive threshold
- [ ] Wired-up `/predict` inference
- [ ] Drift monitoring (PSI)

## How to run it

### 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Get the data

This project uses the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (`mlg-ulb/creditcardfraud`). It isn't committed to this repo.

1. Create a Kaggle account and API token (`kaggle.json`) — see [Kaggle API docs](https://www.kaggle.com/docs/api).
2. Place it at `~/.kaggle/kaggle.json`.
3. Download and place the CSV at `data/raw/creditcard.csv`, e.g.:

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

### 3. Train

```bash
python -m src.models.train
```

### 4. Serve

```bash
uvicorn api.main:app --reload
```

Then open `http://localhost:8000/docs` for interactive Swagger docs, or:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_transaction.json
```

### 5. Test

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
- Shadow-mode A/B testing of the decision threshold before committing to a production cutoff.
- Swap in the IEEE-CIS fraud dataset for richer, non-anonymised feature engineering.
- MLflow for experiment tracking once run volume outgrows flat-file logging.
- Streaming ingestion (Kafka) for near-real-time scoring instead of request/response batch scoring.

## Repo structure

```
fraud-detection-system/
├── data/               # gitignored — raw/processed data
├── notebooks/          # EDA only, no modelling logic
├── src/
│   ├── data/            # load, preprocess
│   ├── models/          # train, evaluate, predict
│   ├── monitoring/       # drift.py
│   └── config.py        # paths, thresholds, hyperparameters
├── api/                # FastAPI app + Pydantic schemas
├── models/              # gitignored model artifacts
├── tests/
├── experiments/          # logged run metrics (json/csv)
├── Dockerfile
└── docker-compose.yml
```
