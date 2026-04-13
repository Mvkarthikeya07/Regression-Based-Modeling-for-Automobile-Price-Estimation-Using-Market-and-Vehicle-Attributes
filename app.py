from flask import Flask, render_template, request, jsonify
from pathlib import Path
import pandas as pd
import numpy as np
import traceback

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "cars.csv"

app = Flask(__name__, static_folder="static", template_folder="templates")

FALLBACK_MEDIAN = None
DF = None


# -------------------------------
# 🔧 CLEANING FUNCTIONS
# -------------------------------

def clean_number(x):
    try:
        s = str(x).lower()
        s = "".join(ch for ch in s if ch.isdigit() or ch == ".")
        return float(s) if s != "" else 0.0
    except:
        return 0.0


def safe_parse_price(x):
    return clean_number(x)


def safe_parse_kms(x):
    return clean_number(x)


# -------------------------------
# 📂 LOAD DATA
# -------------------------------

def load_data():
    global FALLBACK_MEDIAN

    if not DATA_PATH.exists():
        print("Data not found:", DATA_PATH.resolve())
        return None

    try:
        df = pd.read_csv(DATA_PATH, encoding="utf-8")
    except:
        try:
            df = pd.read_csv(DATA_PATH, encoding="latin1")
        except:
            df = pd.read_csv(DATA_PATH, engine="python", encoding="utf-8")

    # Rename columns
    rename = {}
    if "Price" in df.columns: rename["Price"] = "price"
    if "company" in df.columns: rename["company"] = "manufacturer"
    if "kms_driven" in df.columns: rename["kms_driven"] = "mileage"
    if "fuel_type" in df.columns: rename["fuel_type"] = "fuel"

    df = df.rename(columns=rename)

    # Clean numeric columns
    if "price" in df.columns:
        df["price"] = df["price"].apply(safe_parse_price)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    if "mileage" in df.columns:
        df["mileage"] = df["mileage"].apply(safe_parse_kms)
        df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")

    # 🔥 CLEAN STRING COLUMNS (IMPORTANT FIX)
    if "fuel" in df.columns:
        df["fuel"] = df["fuel"].astype(str)

    if "manufacturer" in df.columns:
        df["manufacturer"] = df["manufacturer"].astype(str)

    # Fallback median
    if "price" in df.columns:
        if not df["price"].dropna().empty:
            FALLBACK_MEDIAN = float(df["price"].median(skipna=True))
        else:
            FALLBACK_MEDIAN = 100000.0

    print("Fallback median:", FALLBACK_MEDIAN)

    return df


# Load data
try:
    DF = load_data()
except Exception as e:
    print("Load error:", e)
    print(traceback.format_exc())
    DF = None


# -------------------------------
# 🌐 ROUTES
# -------------------------------

@app.route("/", methods=["GET"])
def index():
    manufacturers = []
    fuels = []

    if DF is not None:

        # ✅ FIXED manufacturer
        if "manufacturer" in DF.columns:
            manufacturers = sorted(
                DF["manufacturer"]
                .dropna()
                .astype(str)
                .str.lower()
                .str.strip()
                .unique()
                .tolist()
            )[:50]

        # ✅ FIXED fuel (MAIN ERROR FIX)
        if "fuel" in DF.columns:
            fuels = sorted(
                DF["fuel"]
                .dropna()
                .astype(str)
                .str.lower()
                .str.strip()
                .unique()
                .tolist()
            )

    if not manufacturers:
        manufacturers = ["toyota", "maruti", "other"]

    if not fuels:
        fuels = ["petrol", "diesel", "electric"]

    return render_template("index.html", manufacturers=manufacturers, fuels=fuels)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # -------------------------------
        # 📥 INPUT
        # -------------------------------
        if request.form and len(request.form) > 0:
            incoming = request.form.to_dict()
            print("FORM:", incoming)
        else:
            incoming = request.get_json(force=True, silent=True) or {}
            print("JSON:", incoming)

        # -------------------------------
        # 🔍 EXTRACT
        # -------------------------------
        kms_raw = incoming.get("kms_driven", incoming.get("mileage", incoming.get("kms", 0)))
        car_age_raw = incoming.get("car_age", incoming.get("age", 0))
        fuel = str(incoming.get("fuel", "")).strip().lower()
        manufacturer = str(incoming.get("manufacturer", "")).strip().lower()

        # -------------------------------
        # 🔧 CLEAN NUMBERS (FINAL FIX)
        # -------------------------------
        kms = clean_number(kms_raw)
        car_age = clean_number(car_age_raw)

        print("DEBUG:", type(kms), kms, type(car_age), car_age)

        # -------------------------------
        # 💰 PREDICTION
        # -------------------------------
        base = FALLBACK_MEDIAN if FALLBACK_MEDIAN else 100000.0

        age_factor = max(0.3, 1 - 0.05 * min(car_age, 15))
        mileage_factor = max(0.7, 1 - min(kms, 200000) / 200000 * 0.3)

        price = round(base * age_factor * mileage_factor, 2)

        # -------------------------------
        # 📤 OUTPUT
        # -------------------------------
        if request.form and len(request.form) > 0:
            return render_template("result.html", price=price)
        else:
            return jsonify({"predicted_price": price})

    except Exception as e:
        print("ERROR:", e)
        print(traceback.format_exc())

        if request.form and len(request.form) > 0:
            return render_template("result.html", price=None, error="Server error")

        return jsonify({"error": "server error"}), 500


# -------------------------------
# 🚀 RUN
# -------------------------------
if __name__ == "__main__":
    print("Server running at http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
