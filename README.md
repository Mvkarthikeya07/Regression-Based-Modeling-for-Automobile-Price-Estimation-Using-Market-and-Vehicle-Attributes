<div align="center">

<img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Flask-REST%20API-000000?style=for-the-badge&logo=flask&logoColor=white"/>
<img src="https://img.shields.io/badge/Scikit--Learn-RandomForest-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/Published-IJRAR-0A66C2?style=for-the-badge&logo=academia&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge"/>

<br/><br/>

# Regression-Based Automobile Price Estimation
### End-to-End ML System for Used Car Valuation · Flask Web App · AJAX Real-Time Predictions

*A production-style machine learning pipeline that predicts used car market value from vehicle attributes — trained, evaluated, and deployed as a responsive web application.*

<br/>

[Overview](#-overview) · [Features](#-features) · [Architecture](#-architecture) · [ML Pipeline](#-ml-pipeline) · [Screenshots](#-screenshots) · [Installation](#-installation) · [Usage](#-usage) · [API Reference](#-api-reference) · [Publication](#-research-publication) · [Recognition](#-hackathon-recognition)

</div>

---

## Overview

The used-car resale market is shaped by a combination of vehicle attributes — manufacturer reputation, mileage, fuel type, and age all contribute to price in non-linear ways. Manual valuation is inconsistent; this system automates it.

This project implements a **Random Forest regression pipeline** trained on real-world used-car data, wrapped in a **Flask REST backend** and served through a responsive web dashboard. Predictions can be triggered either via a standard form POST or instantly via AJAX — no page reload required.

**What makes this different from a typical ML demo:**

- The training pipeline uses `sklearn.pipeline.Pipeline` with a `ColumnTransformer` — preprocessing and the model are bundled together into a single serialized artifact (`model.pkl`), eliminating training/serving skew
- Log-transform on the target (`log1p(price)`) is applied during training and automatically inverted on prediction via a `WrappedPipeline` class — so the model always outputs prices in original rupee scale
- A statistical fallback (`median × age_factor × mileage_factor`) ensures the app never crashes on unseen input combinations
- Input cleaning handles real-world mess: comma-separated prices (`"3,25,000"`), kilometer strings (`"28,000 kms"`), `"Ask For Price"` entries, and encoding inconsistencies

---

## Features

**Machine Learning**
- Random Forest Regressor with 200 estimators (`n_jobs=-1` for parallel training)
- `log1p` target transformation for improved regression on skewed price distributions
- sklearn `Pipeline` + `ColumnTransformer` combining `StandardScaler`, `SimpleImputer`, and `OneHotEncoder`
- Automatic model evaluation: MAE, RMSE, R²
- Fallback formula for zero-downtime predictions on unseen inputs

**Web Application**
- Dual prediction modes: form-based POST and AJAX JSON (no page reload)
- Dynamic dropdowns populated from actual dataset values (manufacturers, fuel types)
- Error-safe routing — server errors return gracefully to the UI
- Responsive frontend with Bootstrap

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         User Interface                           │
│         Form Input (POST)  │  AJAX JSON Request                  │
└──────────────────────┬─────────────────────┬─────────────────────┘
                       │                     │
                       ▼                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Flask Backend (app.py)                      │
│                                                                  │
│   Route: GET /          → Render dashboard with dynamic dropdowns│
│   Route: POST /predict  → Detect input type (form / JSON)       │
│                           → Clean & normalize inputs             │
│                           → Run prediction logic                 │
│                           → Return rendered page or JSON         │
└──────────────────────────────┬───────────────────────────────────┘
                               │
               ┌───────────────┴─────────────────┐
               ▼                                 ▼
┌──────────────────────────┐      ┌──────────────────────────────┐
│  ML Model (model.pkl)    │      │  Fallback Estimator          │
│                          │      │                              │
│  WrappedPipeline         │      │  price = median              │
│  ├ ColumnTransformer     │      │        × age_factor          │
│  │  ├ StandardScaler     │      │        × mileage_factor      │
│  │  ├ SimpleImputer      │      │                              │
│  │  └ OneHotEncoder      │      │  Used when model.pkl is      │
│  └ RandomForestRegressor │      │  absent or input is unseen   │
│    (200 trees, log-scale)│      │                              │
└──────────────────────────┘      └──────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│                     Prediction Output                            │
│           Form POST → result.html  │  AJAX → JSON response      │
└──────────────────────────────────────────────────────────────────┘
```

---

## ML Pipeline

### Dataset

The dataset (`data/cars.csv`) contains real-world Indian used-car listings with the following columns:

| Column | Raw Name | Description |
|---|---|---|
| `name` | `name` | Car model name (dropped after feature engineering) |
| `manufacturer` | `company` | Manufacturer / brand |
| `year` | `year` | Manufacturing year (converted to `car_age`) |
| `price` | `Price` | Target variable — resale price in INR |
| `mileage` | `kms_driven` | Odometer reading (e.g., `"28,000 kms"`) |
| `fuel` | `fuel_type` | Fuel type (Petrol / Diesel / CNG / Electric) |

### Preprocessing

Raw data requires significant cleaning before modeling:

| Step | Detail |
|---|---|
| Price parsing | Strips commas (`"3,25,000"` → `325000`), drops `"Ask For Price"` rows |
| Mileage parsing | Removes `"kms"` suffix, strips commas, converts to float |
| Age computation | `car_age = current_year − year` |
| Categorical normalization | Lowercased and stripped for `manufacturer` and `fuel` |
| Missing values | Median imputation for numerics, most-frequent for categoricals |
| Outlier handling | Implicit via log-target transformation and IQR-robust RF model |

### Training

```python
# Simplified training flow (train.py)
pipeline = Pipeline([
    ("preprocessor", ColumnTransformer([
        ("num", Pipeline([SimpleImputer(median), StandardScaler()]), numeric_cols),
        ("cat", Pipeline([SimpleImputer(most_frequent), OneHotEncoder(handle_unknown="ignore")]), cat_cols)
    ])),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
])

pipeline.fit(X_train, np.log1p(y_train))      # train on log-scale
joblib.dump(WrappedPipeline(pipeline), "model/model.pkl")  # saves auto-inversion wrapper
```

### Fallback Estimator

When `model.pkl` is absent or input combinations are entirely unseen, the system falls back to:

```python
age_factor     = max(0.3, 1 - 0.05 × min(car_age, 15))
mileage_factor = max(0.7, 1 - min(kms, 200_000) / 200_000 × 0.3)
price          = fallback_median × age_factor × mileage_factor
```

This guarantees the application always returns a reasonable estimate rather than crashing.

---

## Repository Structure

```
CAR-PRICE-PREDICTION/
│
├── app.py                   # Flask server — routing, cleaning, prediction, fallback
├── train.py                 # ML training script — pipeline, evaluation, model export
├── requirements.txt         # Python dependencies
│
├── data/
│   └── cars.csv             # Used-car dataset (name, company, year, Price, kms_driven, fuel_type)
│
├── model/
│   └── model.pkl            # Serialized WrappedPipeline (preprocessor + RF + log-inverse)
│
├── static/
│   ├── css/
│   │   └── styles.css       # Dashboard styling
│   └── js/
│       └── app.js           # AJAX prediction logic
│
└── templates/
    ├── index.html           # Input form with dynamic dropdowns
    └── result.html          # Prediction result display
```

---

## Screenshots

### Prediction Interface
![Prediction Interface](https://github.com/user-attachments/assets/52b0948f-e2e4-4f34-8d78-c7d9d3a63a3d)

### Prediction Result
![Prediction Result](https://github.com/user-attachments/assets/244ab290-5361-48e3-ad6b-b05147c2cb31)

---

## Installation

**Prerequisites:** Python 3.8+, pip

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/AI-Car-Price-Prediction.git
cd AI-Car-Price-Prediction
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Usage

### Train the Model

```bash
python train.py
```

Outputs MAE, RMSE, and R² on the held-out test set. Saves `model/model.pkl`. Training uses an 80/20 split with `random_state=42` for reproducibility.

### Run the Web Application

```bash
python app.py
```

Open your browser at `http://127.0.0.1:5000/`. The dashboard loads with dropdowns populated dynamically from the dataset.

> If `model.pkl` is not present, the app automatically falls back to the statistical estimator — no crash, no setup friction.

---

## API Reference

The backend exposes two endpoints:

### `GET /`
Returns the prediction dashboard. Dropdowns are populated with unique manufacturer and fuel values from `cars.csv`.

### `POST /predict`

Accepts both `application/x-www-form-urlencoded` (form) and `application/json` (AJAX).

**Request fields:**

| Field | Type | Description |
|---|---|---|
| `manufacturer` | string | Car manufacturer (e.g., `"maruti"`, `"hyundai"`) |
| `kms_driven` | number/string | Odometer reading in km |
| `fuel` | string | Fuel type (`"petrol"`, `"diesel"`, `"cng"`) |
| `car_age` | number | Age of the vehicle in years |

**AJAX example:**

```javascript
const response = await fetch('/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    manufacturer: 'maruti',
    kms_driven: 45000,
    fuel: 'petrol',
    car_age: 6
  })
});
const data = await response.json();
console.log(data.predicted_price);   // e.g., 285000.0
```

**Python example:**

```python
import requests

resp = requests.post('http://127.0.0.1:5000/predict', json={
    'manufacturer': 'hyundai',
    'kms_driven': 30000,
    'fuel': 'diesel',
    'car_age': 4
})
print(resp.json())   # {"predicted_price": 412000.0}
```

---

## Research Publication

The research work underlying this project has been peer-reviewed and published.

| | |
|---|---|
| **Title** | Machine Learning–Based Automobile Price Prediction System |
| **Journal** | International Journal of Research and Analytical Reviews (IJRAR) |
| **Link** | [ijrar.org/viewfull.php?p_id=IJRAR25D2970](https://www.ijrar.org/viewfull.php?p_id=IJRAR25D2970) |

The paper covers dataset analysis, feature engineering methodology, and model evaluation in detail.

---

## Hackathon Recognition

This project was submitted to **Codegeist 2025: Atlassian Williams Racing Edition** — one of the world's largest developer hackathons, organized by Atlassian.

The submission was among the **first 300 entries worldwide**, earning official Codegeist participant recognition.

📄 [View Official Confirmation from Atlassian (PDF)](https://drive.google.com/file/d/1bRdTqv6edJupHsh3e62eGCW5duk2dTNd/view?usp=drivesdk)

---

## Internship Context

Developed during an AI/ML Internship at **InternPe** (Nov 24 – Dec 21, 2025), with a focus on practical machine learning system design, real-world data preprocessing, and Flask-based ML deployment.

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.8 | 3.10+ |
| RAM | 2 GB | 4 GB+ |
| OS | Windows / Linux / macOS | Ubuntu 20.04+ |

---

## Roadmap

- [ ] Swap fallback estimator with a production-grade model serving layer
- [ ] Add confidence intervals to predictions (RF quantile regression)
- [ ] Extend features: transmission type, number of owners, city of registration
- [ ] Docker containerization for one-command deployment
- [ ] Model versioning and experiment tracking (MLflow)
- [ ] Batch prediction API endpoint for bulk CSV uploads

---

## Author

**M V Karthikeya**
B.Tech — Computer Science (AI & ML)
SRM Institute of Science and Technology

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
  <sub>Built with precision · Peer-reviewed · Hackathon recognized.</sub>
</div>
