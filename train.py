# train.py
"""
Training script adapted to your uploaded dataset:
columns detected: name, company, year, Price, kms_driven, fuel_type

Saves a sklearn Pipeline (preprocessor + regressor) to model/model.pkl
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
import sys

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "cars.csv"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "model.pkl"
RANDOM_STATE = 42

USE_LOG_TARGET = True  # train on log1p(price) and save a wrapper that outputs original scale


def load_raw(path=DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path.resolve()}")
    df = pd.read_csv(path)
    if df.shape[0] == 0:
        raise ValueError("Dataset is empty.")
    return df


def clean_price(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s.lower().startswith("ask") or s.lower().startswith("contact") or s.lower() == "":
        return np.nan
    # remove commas and non-digit characters (except dot)
    s2 = "".join(ch for ch in s if ch.isdigit() or ch == ".")
    try:
        return float(s2) if s2 != "" else np.nan
    except:
        return np.nan


def clean_kms(x):
    if pd.isna(x):
        return np.nan
    s = str(x).lower()
    # remove 'kms', 'km', commas and spaces
    for token in ["kms", "km", "km."]:
        s = s.replace(token, "")
    s = s.replace(",", "").strip()
    try:
        return float(s) if s != "" else np.nan
    except:
        return np.nan


def preprocess_df(df: pd.DataFrame):
    df = df.copy()

    # rename known columns to canonical names if present
    rename_map = {}
    if "Price" in df.columns:
        rename_map["Price"] = "price"
    if "company" in df.columns:
        rename_map["company"] = "manufacturer"
    if "kms_driven" in df.columns:
        rename_map["kms_driven"] = "mileage"
    if "fuel_type" in df.columns:
        rename_map["fuel_type"] = "fuel"
    if "year" in df.columns:
        rename_map["year"] = "year"
    if "name" in df.columns:
        rename_map["name"] = "name"
    df = df.rename(columns=rename_map)

    # required after mapping: price, year, manufacturer, mileage, fuel
    required = ["price", "year", "manufacturer", "mileage", "fuel"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns after mapping: {missing}. Found: {list(df.columns)}")

    # clean price and mileage
    df["price"] = df["price"].apply(clean_price)
    df["mileage"] = df["mileage"].apply(clean_kms)

    # drop rows where price is missing (e.g., 'Ask For Price')
    df = df[df["price"].notna()].reset_index(drop=True)
    if df.shape[0] == 0:
        raise ValueError("No rows with numeric price after cleaning. Check 'Price' values.")

    # compute car_age from year
    current_year = datetime.now().year
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["car_age"] = current_year - df["year"]

    # coerce numeric columns
    df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")
    df["car_age"] = pd.to_numeric(df["car_age"], errors="coerce")

    # standardize categorical text
    df["manufacturer"] = df["manufacturer"].astype(str).str.lower().str.strip()
    df["fuel"] = df["fuel"].astype(str).str.lower().str.strip()

    # drop columns we won't use (name, year)
    df = df.drop(columns=["name", "year"], errors="ignore")

    return df


def build_pipeline(df: pd.DataFrame):
    X = df.drop(columns=["price"])
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    # NOTE: use sparse_output=False for newer scikit-learn versions
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ], remainder="drop")

    model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
    return pipeline


def evaluate_original_scale(y_true_raw, y_pred_raw):
    mae = mean_absolute_error(y_true_raw, y_pred_raw)
    rmse = mean_squared_error(y_true_raw, y_pred_raw, squared=False)
    r2 = r2_score(y_true_raw, y_pred_raw)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def main():
    try:
        print(f"Loading data from: {DATA_PATH}")
        df_raw = load_raw()
        print(f"Raw data shape: {df_raw.shape}")

        print("Cleaning and preprocessing dataset...")
        df = preprocess_df(df_raw)
        print(f"Rows after cleaning: {df.shape[0]}")
        print("Sample columns:", list(df.columns))
        # show simple stats (optional)
        print("Price: min/max:", df["price"].min(), df["price"].max())

        X = df.drop(columns=["price"])
        y = df["price"].copy()

        if USE_LOG_TARGET:
            y_trans = np.log1p(y)
        else:
            y_trans = y

        X_train, X_test, y_train, y_test = train_test_split(X, y_trans, test_size=0.2, random_state=RANDOM_STATE)
        print(f"Train rows: {X_train.shape[0]}, Test rows: {X_test.shape[0]}")

        pipeline = build_pipeline(df)
        print("Training pipeline (this may take a few moments)...")
        pipeline.fit(X_train, y_train)

        print("Predicting on test set...")
        y_pred_test = pipeline.predict(X_test)

        if USE_LOG_TARGET:
            # invert log1p to original scale
            y_pred_orig = np.expm1(y_pred_test)
            y_test_orig = np.expm1(y_test)
        else:
            y_pred_orig = y_pred_test
            y_test_orig = y_test

        metrics = evaluate_original_scale(y_test_orig, y_pred_orig)
        print("Evaluation on test (original price scale):")
        print(f"  MAE  : {metrics['mae']:.2f}")
        print(f"  RMSE : {metrics['rmse']:.2f}")
        print(f"  R2   : {metrics['r2']:.4f}")

        # Save pipeline. If training used log-target, wrap predict to invert log1p automatically.
        if USE_LOG_TARGET:
            class WrappedPipeline:
                def __init__(self, pipe):
                    self.pipe = pipe
                def predict(self, X):
                    raw = self.pipe.predict(X)
                    return np.expm1(raw)
                def __getattr__(self, item):
                    return getattr(self.pipe, item)
            wrapped = WrappedPipeline(pipeline)
            joblib.dump(wrapped, MODEL_PATH)
        else:
            joblib.dump(pipeline, MODEL_PATH)

        print(f"Saved trained pipeline to: {MODEL_PATH.resolve()}")
        print("Done.")

    except Exception as e:
        print("Training failed:", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
