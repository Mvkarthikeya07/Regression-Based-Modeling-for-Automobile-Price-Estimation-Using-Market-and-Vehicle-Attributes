# app.py (reads kms_driven from form/JSON and maps to internal 'mileage')
from flask import Flask, render_template, request, jsonify
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import traceback

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "cars.csv"

app = Flask(__name__, static_folder="static", template_folder="templates")

MODEL = None
FALLBACK_MEDIAN = None
DF = None

def safe_parse_price(x):
    try:
        s = str(x)
        s = "".join(ch for ch in s if ch.isdigit() or ch==".")
        return float(s) if s!="" else np.nan
    except:
        return np.nan

def safe_parse_kms(x):
    try:
        s = str(x).lower()
        for token in ["kms","km","km.","kms.","kilometers"]:
            s = s.replace(token,"")
        s = s.replace(",","").strip()
        return float(s) if s!="" else np.nan
    except:
        return np.nan

def load_data():
    """
    Robust CSV load: try default read_csv first (no engine specified),
    then fall back to other encodings. Avoid passing low_memory with engine='python'.
    """
    global FALLBACK_MEDIAN
    if not DATA_PATH.exists():
        print("Data not found at", DATA_PATH.resolve())
        return None

    df = None
    # 1) try default (fast) read
    try:
        df = pd.read_csv(DATA_PATH, encoding="utf-8")
        print("Loaded CSV with encoding='utf-8' (default engine). Rows:", df.shape[0])
    except Exception as e1:
        print("Default read_csv failed:", e1)
        # 2) try latin1
        try:
            df = pd.read_csv(DATA_PATH, encoding="latin1")
            print("Loaded CSV with encoding='latin1'. Rows:", df.shape[0])
        except Exception as e2:
            print("latin1 read_csv failed:", e2)
            # 3) try with engine='python' but without low_memory arg
            try:
                df = pd.read_csv(DATA_PATH, engine="python", encoding="utf-8")
                print("Loaded CSV with engine='python', encoding='utf-8'. Rows:", df.shape[0])
            except Exception as e3:
                print("Failed to read CSV with multiple attempts. Last error:", e3)
                return None

    # rename common fields if present (keep internal column name 'mileage')
    rename = {}
    if "Price" in df.columns: rename["Price"] = "price"
    if "company" in df.columns: rename["company"] = "manufacturer"
    if "kms_driven" in df.columns: rename["kms_driven"] = "mileage"   # dataset may have kms_driven
    if "mileage" in df.columns: rename["mileage"] = "mileage"
    if "fuel_type" in df.columns: rename["fuel_type"] = "fuel"
    df = df.rename(columns=rename)

    # parse columns if present
    if "price" in df.columns:
        df["price"] = df["price"].apply(safe_parse_price)
    if "mileage" in df.columns:
        df["mileage"] = df["mileage"].apply(safe_parse_kms)

    # compute fallback median if price exists
    if "price" in df.columns:
        FALLBACK_MEDIAN = float(df["price"].median(skipna=True)) if not df["price"].dropna().empty else None
    print("Fallback median:", FALLBACK_MEDIAN)

    return df

# load at startup (so UI lists manufacturers)
try:
    DF = load_data()
except Exception as e:
    print("Unexpected error while loading data:", e)
    print(traceback.format_exc())
    DF = None

@app.route("/", methods=["GET"])
def index():
    manufacturers = []
    fuels = []
    if DF is not None:
        if "manufacturer" in DF.columns:
            manufacturers = sorted(DF["manufacturer"].astype(str).str.lower().str.strip().unique().tolist())[:50]
        if "fuel" in DF.columns:
            fuels = sorted(DF["fuel"].astype(str).str.lower().str.strip().unique().tolist())
    if not manufacturers:
        manufacturers = ["toyota","maruti","other"]
    if not fuels:
        fuels = ["petrol","diesel","electric"]
    return render_template("index.html", manufacturers=manufacturers, fuels=fuels)

@app.route("/predict", methods=["POST"])
def predict():
    # This endpoint logs inputs and returns a heuristic fallback price (guaranteed)
    try:
        incoming = None
        if request.form and len(request.form) > 0:
            incoming = request.form.to_dict()
            print("Received FORM:", incoming)
        else:
            incoming = request.get_json(force=True, silent=True) or {}
            print("Received JSON:", incoming)

        # read kms_driven (preferred), with fallbacks to other keys
        kms_raw = incoming.get("kms_driven", incoming.get("mileage", incoming.get("kms", incoming.get("km", 0))))
        fuel = str(incoming.get("fuel", "")).strip().lower()
        manufacturer = str(incoming.get("manufacturer", "")).strip().lower()
        car_age_raw = incoming.get("car_age", incoming.get("age", 0))

        # coerce numeric inputs safely
        try:
            kms = float(str(kms_raw).replace(",","").strip()) if str(kms_raw).strip() != "" else 0.0
        except:
            kms = 0.0
        try:
            car_age = float(str(car_age_raw).strip()) if str(car_age_raw).strip() != "" else 0.0
        except:
            car_age = 0.0

        # internal dataframe expects 'mileage' column name
        row = {
            "manufacturer": manufacturer,
            "mileage": kms,
            "fuel": fuel,
            "car_age": car_age
        }

        # simple heuristic prediction using fallback median (no ML required for guaranteed result)
        if FALLBACK_MEDIAN is None:
            base = 100000.0
        else:
            base = FALLBACK_MEDIAN
        age_factor = max(0.3, 1 - 0.05 * min(car_age, 15))
        mileage_factor = max(0.7, 1 - min(kms, 200000) / 200000 * 0.3)
        price = round(base * age_factor * mileage_factor, 2)

        # return render for form submit, JSON for AJAX
        if request.form and len(request.form) > 0:
            return render_template("result.html", price=price)
        else:
            return jsonify({"predicted_price": price})
    except Exception as e:
        print("Predict endpoint error:", e)
        print(traceback.format_exc())
        if request.form and len(request.form) > 0:
            return render_template("result.html", price=None, error="Server error")
        return jsonify({"error":"server error"}), 500

if __name__ == "__main__":
    print("Starting debug server. data path:", DATA_PATH.resolve())
    app.run(debug=True, host="0.0.0.0", port=5000)
