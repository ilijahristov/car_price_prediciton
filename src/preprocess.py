"""
Preprocessing for the car price model.

This module is imported by BOTH train.py and predict.py so the two can never
drift apart. It is split into two halves:

  fit_preprocessor(df) -> (X, y, params)
      Runs at TRAINING time on the full dataset. Learns every parameter that
      depends on seeing all the data (medians, category lists, column layout,
      scaler statistics) and returns them in `params` so they can be saved
      into the model artifact.

  transform(records, params) -> X
      Runs at SERVING time on one car (or a few). Learns nothing. It applies
      the saved `params` so a single incoming request is encoded exactly the
      same way the training rows were.

The split exists to prevent train/serve skew: a single car has no median and
no category distribution, so anything statistical MUST come from `params`.
"""

import re

import numpy as np
import pandas as pd

# Columns that get one-hot encoded (low cardinality)
LOW_CARD_COLS = [
    "engine_fuel_type",
    "transmission_type",
    "driven_wheels",
    "vehicle_size",
    "vehicle_style",
]

# Columns that are log-transformed because they are right-skewed
LOG_COLS = ["engine_hp", "highway_mpg", "city_mpg", "popularity"]

# The placeholder MSRP found during EDA: appeared 1036 times, every row
# between 1990-2000. Not a real price, it means "never recorded".
MSRP_PLACEHOLDER = 2000


# ---------------------------------------------------------------------------
# Shared, stateless cleaning (identical at train and serve time)
# ---------------------------------------------------------------------------

def _normalize_columns(df):
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    return df


def _fix_highway_mpg(df):
    """
    A single 2017 Audi A6 row had highway_mpg = 354, impossible for a gasoline
    car. Every other A6 from 2015-2017 sits between 29-35, so a decimal point
    was dropped during data entry. Dividing by 10 lands it back in the cluster.
    """
    df = df.copy()
    df["highway_mpg"] = df["highway_mpg"].astype(float)
    bad = df["highway_mpg"] > 200
    df.loc[bad, "highway_mpg"] = df.loc[bad, "highway_mpg"] / 10
    return df


def _build_tag_columns(df, tag_list):
    """
    market_category is a compound string like "luxury,performance". Split it
    into one binary column per individual tag.

    The regex uses comma/string boundaries rather than a plain substring match,
    otherwise 'performance' would also match inside 'high-performance'.

    `tag_list` comes from params so serving produces the exact same columns
    even when an incoming car's category was never seen in training.
    """
    df = df.copy()
    df["market_category"] = df["market_category"].fillna("Unknown")

    for tag in tag_list:
        pattern = rf"(?:^|,){re.escape(tag)}(?:,|$)"
        df[f"tag_{tag}"] = df["market_category"].str.contains(pattern, regex=True).astype(int)

    return df.drop(columns=["market_category"])


def _apply_logs(df):
    df = df.copy()
    for col in LOG_COLS:
        df[f"log_{col}"] = np.log(df[col])
    return df.drop(columns=LOG_COLS)


def _build_dummy_columns(df, category_levels):
    """
    Deterministic one-hot encoding driven by saved category levels.

    pd.get_dummies(drop_first=True) is data-dependent: it creates a column per
    value it happens to see, and drops whichever value sorts first in THAT
    dataframe. Given a single-row request there is only one value per column,
    so it gets dropped and every dummy comes out as 0 -- silently wrong
    predictions, no error raised.

    Here the levels (already excluding the dropped reference level) are learned
    once during fit and stored, so serving always produces the same columns in
    the same order. An unseen category simply leaves all its dummies at 0.
    """
    df = df.copy()
    for col, levels in category_levels.items():
        values = df[col] if col in df.columns else pd.Series([None] * len(df))
        for level in levels:
            df[f"{col}_{level}"] = (values == level).astype(int)
    return df.drop(columns=[c for c in category_levels if c in df.columns])


# ---------------------------------------------------------------------------
# TRAIN: learn parameters and transform the full dataset
# ---------------------------------------------------------------------------

def fit_preprocessor(df):
    """
    Returns (X, y, params).

    X       : (n_samples, n_features) float array, standardized
    y       : (n_samples, 1) log_msrp target
    params  : everything predict.py needs to reproduce this transformation
    """
    df = _normalize_columns(df)

    # --- target: drop placeholder rows, then log ---
    df = df.copy()
    df.loc[df["msrp"] == MSRP_PLACEHOLDER, "msrp"] = np.nan
    df = df.dropna(subset=["msrp"])
    y = np.log(df["msrp"].values).reshape(-1, 1)

    # --- learn: the full tag vocabulary from market_category ---
    categories = df["market_category"].fillna("Unknown")
    tag_list = sorted({tag for entry in categories.str.split(",") for tag in entry})

    # --- learn: engine_hp medians per make/model, plus an electric fallback ---
    hp_medians = (
        df.groupby(["make", "model"])["engine_hp"].median().dropna().to_dict()
    )
    electric_hp_median = float(
        df.loc[df["engine_fuel_type"] == "electric", "engine_hp"].median()
    )
    global_hp_median = float(df["engine_hp"].median())

    # --- learn: medians for the log-transformed numeric columns ---
    # np.log(None) throws, so any request omitting these needs a fallback.
    numeric_medians = {c: float(df[c].median()) for c in LOG_COLS}

    # --- learn: the mode for number_of_doors ---
    doors_mode = float(df["number_of_doors"].mode()[0])

    # --- learn: frequency encoding for the high-cardinality make column ---
    make_freq = df["make"].value_counts().to_dict()

    # --- learn: the category levels for each one-hot column ---
    # The first level is dropped as the reference category (equivalent to
    # drop_first=True) but the choice is made ONCE here and saved, rather than
    # being re-decided from whatever data shows up at serve time.
    category_levels = {}
    for col in LOW_CARD_COLS:
        levels = sorted(df[col].dropna().unique().tolist())
        category_levels[col] = levels[1:]

    params = {
        "tag_list": tag_list,
        "category_levels": category_levels,
        "hp_medians": hp_medians,
        "electric_hp_median": electric_hp_median,
        "global_hp_median": global_hp_median,
        "doors_mode": doors_mode,
        "numeric_medians": numeric_medians,
        "make_freq": make_freq,
        "feature_names": None,   # filled in below
        "X_mean": None,
        "X_std": None,
    }

    X_df = _shared_feature_build(df, params, is_training=True)

    # --- learn: the exact column layout, then the scaler statistics ---
    params["feature_names"] = X_df.columns.tolist()
    X = X_df.values.astype(float)

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0          # constant columns would divide by zero

    params["X_mean"] = X_mean
    params["X_std"] = X_std

    X = (X - X_mean) / X_std

    _assert_clean(X, y)
    return X, y, params


# ---------------------------------------------------------------------------
# SERVE: apply saved parameters to new records
# ---------------------------------------------------------------------------

def transform(records, params):
    """
    `records` is a dict (one car) or a list of dicts. Returns the standardized
    feature matrix, with columns in exactly the order the model was trained on.
    """
    if isinstance(records, dict):
        records = [records]

    df = _normalize_columns(pd.DataFrame(records))

    # Any column the training data had but the request omitted becomes NaN,
    # so the same imputation logic below can fill it.
    X_df = _shared_feature_build(df, params, is_training=False)

    # Force the exact training column layout: missing -> 0, unexpected -> dropped
    X_df = X_df.reindex(columns=params["feature_names"], fill_value=0)

    X = X_df.values.astype(float)
    return (X - params["X_mean"]) / params["X_std"]


# ---------------------------------------------------------------------------
# The feature build both paths share
# ---------------------------------------------------------------------------

def _shared_feature_build(df, params, is_training):
    df = df.copy()

    # --- engine_hp: make/model median, then electric median, then global ---
    def fill_hp(row):
        if pd.notna(row.get("engine_hp")):
            return row["engine_hp"]
        key = (row.get("make"), row.get("model"))
        if key in params["hp_medians"]:
            return params["hp_medians"][key]
        if row.get("engine_fuel_type") == "electric":
            return params["electric_hp_median"]
        return params["global_hp_median"]

    df["engine_hp"] = df.apply(fill_hp, axis=1)

    # --- engine_cylinders: 0 is literally correct for rotary and electric ---
    df["engine_cylinders"] = df["engine_cylinders"].fillna(0)

    # --- number_of_doors: mode learned at training time ---
    df["number_of_doors"] = df["number_of_doors"].fillna(params["doors_mode"])

    # --- log columns: fill any gaps with the medians learned at fit time ---
    for col, median in params["numeric_medians"].items():
        if col not in df.columns:
            df[col] = median
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(median)

    df = _fix_highway_mpg(df)
    df = _build_tag_columns(df, params["tag_list"])
    df = _apply_logs(df)

    # --- one-hot the low-cardinality categoricals ---
    # NOT pd.get_dummies: that decides which columns exist from whatever values
    # happen to be present, so a single-row request produces different columns
    # than the training set. The levels are learned once and saved instead.
    df = _build_dummy_columns(df, params["category_levels"])

    # --- frequency-encode make, then drop the raw text columns ---
    df["make_freq"] = df["make"].map(params["make_freq"]).fillna(0)
    df = df.drop(columns=["make", "model"], errors="ignore")

    # --- drop the target so it can never leak into the features ---
    df = df.drop(columns=["msrp", "log_msrp"], errors="ignore")

    return df


def _assert_clean(X, y):
    """NaN and inf propagate silently through training instead of crashing."""
    assert not np.isnan(X).any(), "X contains NaN"
    assert not np.isinf(X).any(), "X contains inf (check for log(0))"
    assert not np.isnan(y).any(), "y contains NaN"
    assert not np.isinf(y).any(), "y contains inf (check for log(0))"