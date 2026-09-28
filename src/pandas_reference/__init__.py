"""The original pandas implementation of the data pipeline, kept for reference.

src/data/ and src/monitoring/drift.py were ported to polars; this package
holds the pandas version they were ported from, unchanged apart from
import paths (commit d70d6e9). It is not used by training, evaluation or
the API. It exists so that:

- the two implementations can be read side by side (same module and
  function names: load, preprocess, preprocess_ieee, drift);
- benchmarks/compare_pandas_polars.py can check the polars outputs against
  it and time both versions;
- its original tests (tests/pandas_reference/) keep running in CI.
"""
