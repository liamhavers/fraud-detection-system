"""Check the polars pipeline reproduces the pandas one, then time both.

    python -m benchmarks.compare_pandas_polars parity
    python -m benchmarks.compare_pandas_polars timing --repeats 5

Needs the raw data (README §2) and a dev install (pandas + pyarrow). The
pandas side is src/pandas_reference/, the implementation the polars port
was made from.

`parity` runs both implementations on the real datasets and compares
outputs cell by cell: row order, column order, values (floats to a stated
tolerance), and the fitted IEEE-CIS encoders. Pandas outputs are written to
parquet by a subprocess so the two ~2GB IEEE-CIS feature frames are never
held alongside a second full pipeline run.

`timing` runs each (task, library) pair in its own subprocess, one warm-up
run and then `--repeats` timed runs, and reports the median wall time and
the process's peak RSS (a lifetime high-water mark including setup and
allocator caching, so indicative only). The raw CSVs are in the OS page cache after the
warm-up, so these are compute timings, not cold-disk timings.
"""

import argparse
import json
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from src.data import preprocess as pl_kaggle
from src.data import preprocess_ieee as pl_ieee
from src.data.load import scan_raw_data, scan_raw_ieee_data
from src.monitoring import drift as pl_drift
from src.pandas_reference import drift as pd_drift
from src.pandas_reference import load as pd_load
from src.pandas_reference import preprocess as pd_kaggle
from src.pandas_reference import preprocess_ieee as pd_ieee

FLOAT_TOLERANCE = 1e-9

# --- Tasks: each returns materialised output, so lazy plans are fully executed ---


def pandas_kaggle() -> tuple:
    return pd_kaggle.time_aware_split(pd_kaggle.clean(pd_load.load_raw_data()))


def polars_kaggle() -> list[pl.DataFrame]:
    return pl.collect_all(pl_kaggle.time_aware_split(pl_kaggle.clean(scan_raw_data())))


def pandas_ieee(transaction_df=None, identity_df=None) -> tuple:
    if transaction_df is None:
        transaction_df, identity_df = pd_load.load_raw_ieee_data()
    joined = pd_ieee.join_transaction_identity(transaction_df, identity_df)
    train_df, test_df = pd_ieee.time_aware_split(joined)
    return pd_ieee.engineer_features(train_df, test_df)


def polars_ieee(transaction_lf=None, identity_lf=None) -> list[pl.DataFrame]:
    """Same shape as src/models/train.py: materialise the split, then engineer."""
    if transaction_lf is None:
        transaction_lf, identity_lf = scan_raw_ieee_data()
    joined = pl_ieee.join_transaction_identity(transaction_lf, identity_lf)
    train_df, test_df = pl.collect_all(pl_ieee.time_aware_split(joined))
    return pl.collect_all(pl_ieee.engineer_features(train_df.lazy(), test_df.lazy()))


TASKS = {
    "kaggle_pipeline": "Kaggle: read CSV → clean → time-aware split",
    "ieee_pipeline": "IEEE-CIS: read CSVs → join → split → feature engineering",
    "ieee_features_in_memory": "IEEE-CIS: join → split → feature engineering (CSVs pre-loaded)",
    "psi_drift": "PSI drift report, 30 Kaggle features (train vs. test)",
}


def _prepare_and_run(task: str, library: str):
    """Return a zero-arg callable for the timed part of `task`, doing setup outside it."""
    if task == "kaggle_pipeline":
        return pandas_kaggle if library == "pandas" else polars_kaggle
    if task == "ieee_pipeline":
        return pandas_ieee if library == "pandas" else polars_ieee
    if task == "ieee_features_in_memory":
        if library == "pandas":
            transaction_df, identity_df = pd_load.load_raw_ieee_data()
            return lambda: pandas_ieee(transaction_df, identity_df)
        transaction_df, identity_df = pl.collect_all(scan_raw_ieee_data())
        return lambda: polars_ieee(transaction_df.lazy(), identity_df.lazy())
    if task == "psi_drift":
        train_df, test_df = (df.drop("Class") for df in polars_kaggle())
        if library == "pandas":
            ref_pd, cur_pd = train_df.to_pandas(), test_df.to_pandas()
            return lambda: pd_drift.check_drift(ref_pd, cur_pd)
        return lambda: pl_drift.check_drift(train_df.lazy(), test_df.lazy())
    raise ValueError(task)


def _time_task_in_this_process(task: str, library: str, repeats: int) -> dict:
    run = _prepare_and_run(task, library)
    run()  # warm-up: page cache, imports, polars thread pool
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        timings.append(time.perf_counter() - start)
    return {
        "median_s": statistics.median(timings),
        "min_s": min(timings),
        "max_s": max(timings),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


def timing(repeats: int) -> None:
    print(f"python {platform.python_version()}, polars {pl.__version__}, "
          f"pandas {pd.__version__}, {repeats} timed runs after 1 warm-up\n")
    rows = []
    for task, description in TASKS.items():
        result = {}
        for library in ("pandas", "polars"):
            completed = subprocess.run(
                [sys.executable, "-m", "benchmarks.compare_pandas_polars",
                 "_time", task, library, str(repeats)],
                capture_output=True, text=True, check=True,
            )
            result[library] = json.loads(completed.stdout.strip().splitlines()[-1])
        speedup = result["pandas"]["median_s"] / result["polars"]["median_s"]
        rows.append((description, result, speedup))
        print(f"{description}\n  pandas {result['pandas']}\n  polars {result['polars']}\n"
              f"  speedup {speedup:.1f}x\n")

    print("| Stage | pandas (median) | polars (median) | Speedup |")
    print("|---|---:|---:|---:|")
    for description, result, speedup in rows:
        print(f"| {description} | {result['pandas']['median_s']:.2f}s "
              f"| {result['polars']['median_s']:.2f}s | {speedup:.1f}× |")


# --- Parity ---


def _compare_frames(name: str, expected: pl.DataFrame, actual: pl.DataFrame) -> bool:
    """Print a cell-level comparison; return True if the frames match."""
    ok = True
    print(f"[{name}] rows: pandas {expected.height:,} / polars {actual.height:,}")
    if expected.height != actual.height:
        return False

    only_expected = [c for c in expected.columns if c not in actual.columns]
    only_actual = [c for c in actual.columns if c not in expected.columns]
    if only_expected or only_actual:
        print(f"  columns only in pandas: {only_expected}, only in polars: {only_actual}")
        for pd_col in only_expected:
            twins = [c for c in only_actual if expected[pd_col].equals(actual[c], null_equal=True)]
            print(f"    pandas {pd_col!r} is value-identical to polars {twins or 'nothing'}")
            ok &= bool(twins)
    shared = [c for c in expected.columns if c in actual.columns]
    same_order = shared == [c for c in actual.columns if c in expected.columns]
    print(f"  {len(shared)} shared columns, same relative order: {same_order}")
    ok &= same_order

    worst_float_diff, dtype_changes, mismatched = 0.0, set(), []
    for col in shared:
        e, a = expected[col], actual[col]
        if e.dtype != a.dtype:
            dtype_changes.add(f"{e.dtype}→{a.dtype}")
        if e.dtype.is_numeric() and a.dtype.is_numeric():
            e, a = e.cast(pl.Float64), a.cast(pl.Float64)
            if not (e.is_null() == a.is_null()).all():
                mismatched.append(col)
                continue
            diff = (e - a).abs().max()
            worst_float_diff = max(worst_float_diff, diff or 0.0)
            if diff is not None and diff > FLOAT_TOLERANCE:
                mismatched.append(col)
        elif not e.equals(a, null_equal=True):
            mismatched.append(col)
    print(f"  columns with differing values: {mismatched or 'none'}")
    print(f"  max |float difference|: {worst_float_diff:.3g} (tolerance {FLOAT_TOLERANCE:g})")
    if dtype_changes:
        print(f"  dtype changes (values compared as Float64): {sorted(dtype_changes)}")
    return ok and not mismatched


def _pandas_ieee_to_parquet(out_dir: str) -> None:
    transaction_df, identity_df = pd_load.load_raw_ieee_data()
    joined = pd_ieee.join_transaction_identity(transaction_df, identity_df)
    del transaction_df
    train_df, test_df = pd_ieee.time_aware_split(joined)
    encoders = pd_ieee.fit_feature_encoders(train_df)
    train_fe, test_fe = pd_ieee.engineer_features(train_df, test_df)
    train_fe.to_parquet(Path(out_dir) / "train_fe.parquet", index=False)
    test_fe.to_parquet(Path(out_dir) / "test_fe.parquet", index=False)
    (Path(out_dir) / "encoders.json").write_text(json.dumps({
        "category_maps": encoders["category_maps"],
        "missing_indicator_columns": encoders["missing_indicator_columns"],
        "card1_frequency": encoders["card1_frequency"].sort_index().to_dict(),
        "card1_mean_amount": encoders["card1_mean_amount"].sort_index().to_dict(),
        "global_mean_amount": encoders["global_mean_amount"],
    }, default=str))


def parity() -> None:
    results = {}

    print("=== Kaggle: clean + time-aware split ===")
    pd_train, pd_test = pandas_kaggle()
    pl_train, pl_test = polars_kaggle()
    results["kaggle train"] = _compare_frames("train", pl.from_pandas(pd_train), pl_train)
    results["kaggle test"] = _compare_frames("test", pl.from_pandas(pd_test), pl_test)

    print("\n=== PSI drift report (Kaggle train vs. test, and a synthetic drifted window) ===")
    ref_pd, cur_pd = pd_train.drop(columns=["Class"]), pd_test.drop(columns=["Class"])
    drifted_pd = cur_pd.assign(Amount=cur_pd["Amount"] * 3 + 50)
    ref_pl, cur_pl = pl_train.drop("Class").lazy(), pl_test.drop("Class").lazy()
    drifted_pl = cur_pl.with_columns(pl.col("Amount") * 3 + 50)
    for label, cur_p, cur_l in [("train vs test", cur_pd, cur_pl),
                                ("train vs drifted", drifted_pd, drifted_pl)]:
        report = pl.from_pandas(pd_drift.check_drift(ref_pd, cur_p)).join(
            pl_drift.check_drift(ref_pl, cur_l), on="feature", suffix="_polars")
        max_diff = (report["psi"] - report["psi_polars"]).abs().max()
        flags_match = report["drifted"].equals(report["drifted_polars"])
        print(f"[{label}] {report.height} features, max |PSI difference| {max_diff:.3g}, "
              f"drift flags identical: {flags_match}")
        results[f"psi {label}"] = max_diff <= FLOAT_TOLERANCE and flags_match
    rng = np.random.default_rng(0)
    array_cases = {
        "normal vs normal": (rng.normal(size=5000), rng.normal(size=5000)),
        "normal vs shifted": (rng.normal(size=5000), rng.normal(loc=5, size=5000)),
        "heavy ties": (rng.integers(0, 4, 5000), rng.integers(0, 6, 5000)),
        "constant reference": (np.ones(100), np.ones(100) * 2),
    }
    for label, (ref, cur) in array_cases.items():
        expected = pd_drift.population_stability_index(ref, cur)
        actual = pl_drift.population_stability_index(ref, cur)
        print(f"[population_stability_index, {label}] pandas {expected:.12f} "
              f"polars {actual:.12f}")
        results[f"psi {label}"] = abs(expected - actual) <= FLOAT_TOLERANCE
    del pd_train, pd_test, ref_pd, cur_pd, drifted_pd

    print("\n=== IEEE-CIS: join + split + fit encoders + feature engineering ===")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, "-m", "benchmarks.compare_pandas_polars",
                        "_pandas_ieee", tmp], check=True)
        expected = json.loads((Path(tmp) / "encoders.json").read_text())

        transaction_lf, identity_lf = scan_raw_ieee_data()
        joined = pl_ieee.join_transaction_identity(transaction_lf, identity_lf)
        train_df, test_df = pl.collect_all(pl_ieee.time_aware_split(joined))
        encoders = pl_ieee.fit_feature_encoders(train_df.lazy())

        stats = encoders["card1_stats"]
        card1_keys = [str(k) for k in stats["card1"]]
        encoder_checks = {
            "category vocabularies + codes": encoders["category_maps"] == expected["category_maps"],
            "card1_frequency": dict(zip(card1_keys, stats["card1_frequency"]))
            == expected["card1_frequency"],
            "card1_mean_amount": np.allclose(
                stats["card1_mean_amount"].to_numpy(),
                [expected["card1_mean_amount"][k] for k in card1_keys], rtol=0, atol=1e-9),
            "global_mean_amount": abs(encoders["global_mean_amount"]
                                      - expected["global_mean_amount"]) <= 1e-9,
        }
        for check, passed in encoder_checks.items():
            print(f"[encoders] {check}: {'match' if passed else 'MISMATCH'}")
            results[f"ieee encoders {check}"] = passed
        print(f"[encoders] missing-indicator columns\n"
              f"  pandas: {expected['missing_indicator_columns']}\n"
              f"  polars: {encoders['missing_indicator_columns']}")

        train_fe, test_fe = pl.collect_all([
            pl_ieee.transform_features(train_df.lazy(), encoders),
            pl_ieee.transform_features(test_df.lazy(), encoders),
        ])
        del train_df, test_df
        for split, actual in [("train_fe", train_fe), ("test_fe", test_fe)]:
            results[f"ieee {split}"] = _compare_frames(
                split, pl.read_parquet(Path(tmp) / f"{split}.parquet"), actual)

    print("\n=== Summary ===")
    for check, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {check}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    if sys.argv[1] == "_time":
        _, _, task, library, repeats = sys.argv
        print(json.dumps(_time_task_in_this_process(task, library, int(repeats))))
    elif sys.argv[1] == "_pandas_ieee":
        _pandas_ieee_to_parquet(sys.argv[2])
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("mode", choices=["parity", "timing"])
        parser.add_argument("--repeats", type=int, default=5)
        args = parser.parse_args()
        parity() if args.mode == "parity" else timing(args.repeats)
