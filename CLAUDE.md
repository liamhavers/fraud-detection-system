# Fraud Detection System — Project Brief for Claude Code

> **Phases 0–7 are complete** (see README "Status"). Phase 7 (validation split and probability calibration) finished on 2026-11-07; its spec is kept at the bottom of this file in [Phase 7](#phase-7--validation-split-and-probability-calibration-complete) for reference. Step 8 (serve the calibrated model) is also done: the API serves unweighted XGBoost + Platt from the `model-v2` release at the cost-derived threshold. The optional IEEE-CIS calibration stretch was explicitly declined by Liam; don't start it.

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

---

## Phase 7 — Validation split and probability calibration (complete)

**Status: Steps 1–8 done (2026-11-07). IEEE-CIS stretch declined by Liam.** Step 8 served unweighted + Platt (Liam's choice: the validation winner), with `train.py` now saving each model's calibrator alongside it and the Dockerfile downloading the model + calibrator pair from `model-v2` if either is missing. Decisions made with Liam along the way: no refit on train + validation; merged middle reliability bins (`CALIBRATION_BIN_EDGES`); calibration method chosen by 5-fold CV log loss within validation (Platt for all three models, `CALIBRATION_METHOD`); the A/B test switched to a paired bootstrap (the original resampled the two policies independently); a second A/B comparison, best calibrated policy at the analytic threshold vs raw class-weighted. Results and which hypotheses held are in README "Phase 7".

### Why this phase exists

Two weaknesses came out of reviewing the finished project. Both matter for Liam's current applications, especially credit-risk Data Scientist roles where a model's probabilities feed pricing and expected loss directly, and where "calibration as a first-order concern" is an explicit requirement.

1. **The test set is used for selection.** `ab_test.py` picks the cost-minimising threshold on `y_test` and then reports cost on the same `y_test`. The comment in `ab_test.py` and the caveat in README Phase 3 already admit this. The SMOTE vs class-weighting choice was also made on test PR-AUC. The PR-AUC numbers are fair estimates, because nothing was tuned on test to produce them. The threshold, the expected cost and the model choice are optimistic.
2. **The served probabilities aren't probabilities.** `scale_pos_weight` (about 578:1) and SMOTE both teach the model a much higher fraud base rate than the real 0.17%. Scores are inflated, which is why the class-weighted cost-optimal threshold is 0.29 and SMOTE's is 0.88. For a calibrated model, decision theory gives the threshold directly: t* = C_FP / (C_FP + C_FN) = 5 / 105 ≈ 0.048.

The goal of this phase is to fix the first weakness, then measure and fix the second, and write both up honestly. That includes the results that don't come out as predicted.

### How to work with Liam on this phase

- **This is a learning exercise as much as a code change.** Liam will be asked about every choice here in interviews, without AI help, so he has to be able to defend each one unaided. Before each step, explain in a few sentences what you're about to do and why, then build it.
- **Stop and ask at decision points** rather than deciding silently. The decision points are marked **[ASK LIAM]** below.
- **Keep it simple.** Prefer small, typed functions that are easy to read. Where a method has a simple closed form (Platt scaling, calibration slope), implement it directly so the parameters are visible, rather than hiding it behind a wrapper.
- **Test against synthetic data with a known truth.** Each new statistical function gets a test on synthetic data where the right answer is known in advance, the same style as `test_ab_test.py`.
- **Commit after each step** with a clear message, and run the full test suite (`pytest`) before each commit.

### Hard boundaries for this phase (do not touch)

- `time_aware_split` in `preprocess.py` and `preprocess_ieee.py`: keep it unchanged. Drift monitoring uses it for its reference and current windows, and the pandas parity work in Phase 6 depends on it. Add a new function next to it instead.
- `src/pandas_reference/` and `tests/pandas_reference/`: leave as they are.
- The IEEE-CIS pipeline. Nothing is tuned on its test set (only PR-AUC is reported), so its evaluation isn't optimistic. See the optional stretch at the end.
- The API, `predict.py`, the Dockerfile, the `model-v1` release artifact and the live Render demo. Changing the served model is a separate, optional step (Step 8) that needs Liam's go-ahead.
- `XGB_PARAMS`. Keep the hyperparameters fixed, so the only thing that changes is the split and calibration. No early stopping and no tuning in this phase.

### Step 1 — Three-way time-ordered split (Kaggle only)

- **`config.py`**
  - Add `VAL_SIZE = 0.2` as a fraction of the full dataset.
  - Keep `TEST_SIZE = 0.2`, giving a 60 / 20 / 20 split.
  - Comment why: the validation slice is taken from what was previously training data, so the test slice (the last 20%, 56,746 transactions, 74 frauds) is identical to before, and old and new test numbers are directly comparable.
- **`src/data/preprocess.py`**: add `time_aware_train_val_test_split(lf, val_size=VAL_SIZE, test_size=TEST_SIZE) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]`.
  - Use the same approach as `time_aware_split`: a stable sort on `Time`, a row index, and boundaries rounded down.
  - Boundaries: `floor(n * (1 - test_size - val_size))` and `floor(n * (1 - test_size))`.
  - Raise `ValueError` unless `0 < val_size`, `0 < test_size` and `val_size + test_size < 1`.
- **Tests (`tests/test_preprocess.py`)**
  - Rows are time-ordered across the three slices.
  - No overlap between slices, and together they cover every row.
  - Slice sizes are correct, including rounding down at the boundaries.
  - The new test slice is row-for-row identical to `time_aware_split`'s test slice. This is the property that makes the before/after comparison valid.
  - Tied timestamps keep their original order.
  - Invalid sizes raise.
- **Log the fraud count in each slice.** This matters: calibration and threshold choice on validation rest on roughly 70 to 100 frauds, which limits how precise they can be. Tell Liam the actual number.

### Step 2 — Training on train only, comparing on validation

In `train.py`, `_run_kaggle_training`:

- Use the three-way split. Fit every model on the train slice only.
- **Add an unweighted XGBoost** (same `XGB_PARAMS`, no `scale_pos_weight`, no SMOTE) as the calibration control. Add `MODEL_ARTIFACT_UNWEIGHTED_FILE = MODELS_DIR / "model_unweighted.pkl"` to config and save it. Log loss is a proper scoring rule, so this model should come out roughly calibrated without any correction. That's the hypothesis to check.
- **Report PR-AUC on validation only** for the baseline logistic regression and the three XGBoost variants. Model comparison now happens on validation.
- **Don't compute test metrics in `train.py` any more.** Test is touched once, in the Step 4 report.
- Log n and fraud count per slice in the experiment JSON.
- Note: retraining overwrites the local `model_class_weighted.pkl`. The live API is unaffected, because Docker and Render download the `model-v1` release asset. Don't change that.
- **[ASK LIAM]** Explain the trade-off: production teams often refit on train plus validation after selection to use the most recent data, at the cost of the calibrator no longer matching the refitted model exactly. Recommend not refitting here (simpler, and the calibrator stays valid), and record the choice in the README.

### Step 3 — Calibration and calibration metrics (`src/models/calibrate.py`)

Use scikit-learn primitives. Don't bump the scikit-learn version for this phase.

**Calibrators** (fit on validation only)

- **Platt scaling, implemented directly.**
  - Take the model's log-odds score: use `model.predict(X, output_margin=True)` for XGBoost. That's cleaner than taking the logit of a clipped probability.
  - Fit an unpenalised logistic regression of `y_val` on that score: `LogisticRegression(penalty=None)`. The default L2 penalty would shrink the parameters.
  - The result is `logit(p_cal) = a * score + b`. Return `a` and `b` explicitly so they can be reported and explained.
  - For the baseline logistic regression pipeline, use its decision function, or skip calibrating it if that's awkward. Calibrating the three XGBoost variants is what matters.
- **Isotonic regression**: `IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)` fitted on the predicted probability.
- **Wrap both in a small `Calibrator` dataclass** (method name, fitted parameters or object, and `predict_proba(model, X) -> np.ndarray`), and save each as a joblib artifact under `models/`.

**Metrics**

| Function | Notes |
|---|---|
| `brier_score(y, p)` | Use `sklearn.metrics.brier_score_loss`. |
| `brier_skill_score(y, p)` | 1 − Brier / (π(1−π)), where π is the slice's observed rate, so the Brier score is read against "always predict the base rate". |
| `log_loss_score(y, p)` | Clip p to [1e-15, 1 − 1e-15]. |
| `calibration_in_the_large(y, p)` | Return mean predicted, observed rate, and their ratio (observed / expected, O/E). |
| `calibration_slope_intercept(y, p)` | Unpenalised logistic regression of y on logit(p) (clipped). Slope below 1 means overconfident, above 1 underconfident; intercept away from 0 means a shifted curve. Return both. |
| `reliability_table(y, p, bin_edges)` | Per bin: n, frauds, mean predicted, observed rate, and a **Wilson score interval** for the observed rate. Implement Wilson directly (a few lines) rather than adding statsmodels. |
| `expected_calibration_error(table)` | n-weighted mean of the absolute gap per bin. Report it as secondary, because it depends on the binning. |

**Binning decision**

At a 0.17% base rate, equal-width or plain quantile bins put almost all the test frauds in one bin. Use **log-spaced bin edges on predicted probability** as the default. For example: `[0, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]`, kept in config as `CALIBRATION_BIN_EDGES`. Show n and fraud count per bin, so bins resting on a handful of frauds are visible as such. **[ASK LIAM]** Show him the bin counts on validation before fixing the edges.

**Plot**

Add `plot_reliability_diagram(tables_by_label, save_path)` to `evaluate.py`, in the same style as the existing plots (`_SURFACE_COLOR`, `_GRID_COLOR`, colours by policy):
- Log-scaled axes and a dashed diagonal.
- Wilson error bars.
- A small score histogram underneath.
- Bins with zero frauds can't go on a log y-axis. Mark them at the bottom edge rather than dropping them silently.

**Threshold helpers**

- Add `analytic_threshold(fn_cost=COST_FALSE_NEGATIVE, fp_cost=COST_FALSE_POSITIVE) -> float`, returning `fp_cost / (fp_cost + fn_cost)`. Compute it from config, never hard-code it.
- **Important fix:** `select_cost_minimising_threshold` searches `np.linspace(0.01, 0.99, 99)`. That grid is too coarse for calibrated probabilities, where the optimum sits near 0.048. Add an optional `thresholds` argument, with the current grid as the default so existing behaviour and tests are unchanged. For calibrated scores, pass a fine log-spaced grid such as `np.geomspace(1e-4, 0.99, 500)`.

**Tests (`tests/test_calibrate.py`)**, all on synthetic data with a known truth:

- **Inflated scores.** Generate a true probability `p_true` at a low base rate, draw `y ~ Bernoulli(p_true)`, and inflate the odds by a known `w` (logit + log w). Platt should recover `a ≈ 1` and `b ≈ −log(w)` within tolerance, and calibrated O/E should come out near 1.
- **Perfectly calibrated input.** Slope ≈ 1, intercept ≈ 0, O/E ≈ 1.
- **Overconfident input** (logits multiplied by 2). Slope ≈ 0.5.
- **Isotonic output** is monotone non-decreasing in the input score and stays within [0, 1].
- **Order preservation.** Platt leaves PR-AUC exactly unchanged, because it's a strictly monotone transform.
- **Wilson interval** contains the true rate in about 95% of simulated draws (loose tolerance), and handles 0 and n successes without errors.
- **Brier skill score** is 0 when every prediction equals the base rate.
- **`analytic_threshold()`** returns 5/105 with the config costs.

### Step 4 — Calibration report (`python -m src.models.calibration_report`)

A new script. Test is touched here, once, for final reporting.

1. Load the three-way split, the three XGBoost artifacts, and fit Platt and isotonic calibrators for each on validation.
2. For each of the 9 combinations (unweighted, class-weighted, SMOTE × raw, Platt, isotonic), compute on **test**:
   - PR-AUC
   - Brier score and Brier skill score
   - log loss
   - O/E
   - calibration slope and intercept
   - ECE
3. Compare thresholds, each evaluated by expected cost on test:
   - **(a)** threshold searched on **validation** (use the fine grid for calibrated scores, the original grid for raw)
   - **(b)** the **analytic** threshold of about 0.048 (only meaningful for calibrated scores, but report it for raw too, to show what goes wrong)
   - **(c)** an **oracle** threshold searched on test, clearly labelled as reference only. It reproduces the old, optimistic approach, so the gap between (a) and (c) measures how optimistic the old numbers were.

   For each, report cost reduction against the naive "never flag" baseline (7,400 on this test set).
4. Save outputs:
   - `reports/reliability_raw.png` (three raw models)
   - `reports/reliability_calibrated.png` (each model's chosen calibrator)
   - full tables via `log_experiment("calibration_report", ...)`
   - a printed markdown table that can be pasted into the README
5. Print the fitted Platt `a` and `b` for each model next to the prior-shift prediction:
   - class-weighted: b ≈ −log(w), where w = n_neg / n_pos on the train slice
   - SMOTE: b ≈ log(π / (1 − π)), because SMOTE balances the training data to 50/50

   Real trees saturate near 0 and 1, so expect deviation, and explain it rather than hide it.

### Step 5 — Re-run the shadow-mode A/B test without test-set leakage

- In `ab_test.py`'s `__main__`, select each policy's threshold on validation, then compute cost, confusion matrices and the bootstrap comparison on test.
- Delete the "deliberate simplification" comment, because it's no longer true.
- Keep the comparison as raw class-weighted vs raw SMOTE, so it stays the same experiment with only the leakage removed.
- **[ASK LIAM]** Whether to also compare the best calibrated policy at the analytic threshold against raw class-weighted at its validation threshold, using the existing `bootstrap_cost_difference`.

### Step 6 — Notebook `notebooks/03_calibration.ipynb`

A narrative walk-through for a human reader, in this order:

1. Why calibration is separate from PR-AUC: ranking vs level.
2. The raw reliability diagram.
3. The metrics table.
4. The Platt `a` and `b` vs the prior-shift prediction.
5. The threshold story: 0.29 / 0.88 → about 0.048, and the validation vs oracle cost gap.
6. Platt vs isotonic, given roughly 70 to 100 validation frauds.
7. Limitations.

Import the logic from `src/`. Don't re-implement it in cells.

### Step 7 — README and CLAUDE.md updates

- **New README section, "Phase 7 — Validation split & probability calibration"**, covering:
  - what was wrong
  - what changed
  - the results table and the two reliability diagrams
  - the threshold result
  - an honest limitations paragraph: small fraud counts in validation and test; a dataset spanning about 48 hours, so "time-ordered" means hours, not months; and the calibrator fitted on a single validation window
- **Update the README Phase 2 and Phase 3 tables to the new numbers.** Keep one line noting the old figures (PR-AUC 0.798, threshold 0.29, chosen on test) and why they changed. Remove the "simplification worth flagging" caveat in Phase 3, because it's now fixed.
- Add calibration to "Key Design Principles" in one or two sentences, and update "How to Run It" with the new command.
- **State the hypotheses below and whether each held.** If a prediction failed, say so plainly; that's worth more than a clean story.
- Tick off this phase in "Status" and in this file.

### Step 8 (optional, only with Liam's explicit go-ahead) — Serve the calibrated model

- `FraudPredictor` would apply the chosen calibrator and use `DECISION_THRESHOLD = analytic_threshold()`. `fraud_probability` in the API response would then be a real probability, which is the point.
- This needs:
  - a new release (`model-v2`) with the model plus calibrator artifacts
  - Dockerfile fallback URLs updated to `model-v2`
  - `test_api.py` / `test_predict.py` updated
  - re-verification of the live Render demo with `sample_transaction.json`
- Don't start this step without asking.

### Hypotheses to check (predictions, not facts)

1. **PR-AUC.** Platt leaves PR-AUC exactly unchanged. Isotonic may shift it slightly, because its steps create tied scores.
2. **Unweighted model.** It is roughly calibrated raw: slope near 1, O/E near 1.
3. **Class-weighted and SMOTE, raw.** Predicted far exceeds observed (O/E well below 1), with much worse Brier score and log loss than unweighted. Calibration recovers most of the gap.
4. **Thresholds.** After calibration, the analytic threshold of about 0.048 gives a test cost close to the validation-searched threshold. The old 0.29 and 0.88 operating points were artefacts of inflated scores.
5. **Platt vs isotonic.** With only about 70 to 100 validation frauds, Platt beats isotonic on test log loss (isotonic overfits small samples).
6. **Headline PR-AUC.** Test PR-AUC drops somewhat from 0.798, because training data shrinks from 80% to 60% and the most recent pre-test data moves into validation.
7. **Optimism of the old approach.** The oracle (test-searched) threshold beats the validation-searched one by a visible margin. That margin is the optimism in the old Phase 3 numbers.

### Known issue: note it, don't fix it in this phase

`Time` is used as a model feature (`FEATURE_COLUMNS` in `predict.py`). Under a time-ordered split, test values of `Time` lie outside the training range, so a tree can only learn "when", not "what". The Databricks follow-on project excludes absolute time for exactly this reason. Removing it changes the API schema, so don't do it here. Add a README "Next Steps" line, and mention it to Liam.

### Optional stretch — IEEE-CIS calibration

IEEE-CIS has about 3.5% fraud, so a validation slice holds thousands of frauds and isotonic becomes viable. That makes a useful contrast with Kaggle on when to choose isotonic over Platt. It needs a three-way split there too, with `engineer_features` fitted on train and applied to validation and test. Only do this if Liam asks for it after Steps 1 to 7 are done.

### Suggested commit sequence

1. `Add time-ordered train/validation/test split with tests`
2. `Train on train slice only; add unweighted XGBoost control; compare on validation`
3. `Add calibration module (Platt, isotonic) and calibration metrics with tests`
4. `Add calibration report script and reliability diagrams`
5. `Select A/B test thresholds on validation instead of test`
6. `Add calibration notebook`
7. `Document Phase 7 results in README`
