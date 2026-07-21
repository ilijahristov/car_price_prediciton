"""
Training entrypoint.

Reads the raw CSV, fits the model, writes a single artifact containing both
the fitted model and every preprocessing parameter it depends on. Run offline,
occasionally. The serving layer never runs any of this.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from preprocess import fit_preprocessor, transform

DATA_PATH = Path(__file__).parent.parent / "data" / "data.csv"
MODEL_DIR = Path(__file__).parent.parent / "models"
MSRP_PLACEHOLDER = 2000

PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)


def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.lower().str.replace(" ", "_")

    # Fixed domain rule, not a learned statistic, so it is safe to apply
    # before the split: 2000 is a placeholder meaning "price never recorded".
    df.loc[df["msrp"] == MSRP_PLACEHOLDER, "msrp"] = np.nan
    return df.dropna(subset=["msrp"]).reset_index(drop=True)


def evaluate(model, X, y_log):
    """Report in log space (what the model optimises) and in dollars (what a
    human understands). MAPE matters more than MAE here because being $5k off
    on a $20k car is far worse than $5k off on a $200k car."""
    pred_log = model.predict(X).reshape(-1, 1)
    pred_d, true_d = np.exp(pred_log), np.exp(y_log)
    return {
        "rmse_log": float(np.sqrt(np.mean((pred_log - y_log) ** 2))),
        "mae_usd": float(np.mean(np.abs(pred_d - true_d))),
        "mape_pct": float(np.mean(np.abs((pred_d - true_d) / true_d)) * 100),
    }


def main():
    df = load_data()

    # Split the RAW frame, so the preprocessor never sees a test row.
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    X_train, y_train, params = fit_preprocessor(df_train)
    X_test = transform(df_test, params)
    y_test = np.log(df_test["msrp"].values).reshape(-1, 1)

    # .ravel() matters: sklearn-style estimators expect y as (n,), and a
    # column vector silently produces a much worse fit with some of them.
    model = XGBRegressor(**PARAMS)
    model.fit(X_train, y_train.ravel())

    metrics = {
        "train": evaluate(model, X_train, y_train),
        "test": evaluate(model, X_test, y_test),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": len(params["feature_names"]),
        "model_params": PARAMS,
    }

    print(f"train  MAPE {metrics['train']['mape_pct']:.2f}%  MAE ${metrics['train']['mae_usd']:,.0f}")
    print(f"test   MAPE {metrics['test']['mape_pct']:.2f}%  MAE ${metrics['test']['mae_usd']:,.0f}")

    print("\ntop features by gain:")
    for name, score in sorted(
        zip(params["feature_names"], model.feature_importances_),
        key=lambda kv: kv[1], reverse=True
    )[:8]:
        print(f"  {name:<45} {score:.4f}")

    # Model and params travel together. A model without its preprocessing
    # parameters cannot make a correct prediction.
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump({"model": model, "params": params}, f)
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nsaved {MODEL_DIR / 'model.pkl'}")


if __name__ == "__main__":
    main()