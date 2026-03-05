🚗 Regression-Based Modeling for Automobile Price Estimation Using Market and Vehicle Attributes

<p align="center"> <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python"> <img src="https://img.shields.io/badge/Flask-Web%20Framework-black?style=for-the-badge&logo=flask"> <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge&logo=scikitlearn"> <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge"> </p>
📌 Overview

This project is a complete end-to-end Machine Learning + Web Application, developed during my InternPe Internship.

The application predicts the market value of a used car based on key vehicle attributes using a regression-based Machine Learning model, and delivers real-time predictions through a clean, responsive Flask web interface.

The project demonstrates real-world ML deployment, clean backend architecture, and production-style design principles, making it suitable for academic evaluation, internships, and professional portfolios.

🚀 Key Features
🔍 Machine Learning

Cleans and preprocesses real-world used-car data

Handles both numerical and categorical features

Regression-based car price prediction

Fallback median-based estimation to ensure system stability

Easily extendable to Random Forest, XGBoost, and other advanced models

🌐 Web Application

Flask backend with organized routing

Clean, responsive UI using HTML, CSS, and Bootstrap

Supports:

Traditional Form POST predictions

AJAX-based real-time predictions (no page reload)

Error-safe design for a smooth and reliable user experience

🧠 Machine Learning Workflow
🧹 Data Preprocessing

Converts kilometer strings → numerical values

Standardizes manufacturer names

Normalizes fuel-type labels

Removes outliers for improved prediction stability

🔮 Prediction Logic

Primary ML regression model predicts the car price

If unseen or unstable inputs are encountered, a fallback formula ensures reliability:

predicted_price = median_price × age_factor × kms_factor


This design guarantees robust predictions without system failure, even for edge cases.

🛠 Tech Stack
Layer	Technology
Language	Python
Backend	Flask
Frontend	HTML, CSS, Bootstrap, JavaScript
Machine Learning	NumPy, Pandas, Scikit-learn
Environment	Virtualenv
Internship	InternPe

📂 Project Structure
'''''
CAR-PRICE-PREDICTION/
│
├── app.py                     # Flask backend
├── train.py                   # ML training script
├── requirements.txt           # Dependencies
│
├── data/
│   └── cars.csv               # Dataset
│
├── model/
│   └── model.pkl              # Trained ML model
│
├── static/
│   ├── css/
│   │   └── styles.css         # Custom styling
│   └── js/
│       └── app.js             # AJAX prediction logic
│
└── templates/
    ├── index.html             # Main user interface
    └── result.html            # Prediction results page
''''''

⚙️ Installation & Setup

1️⃣ Clone the Repository
git clone https://github.com/your-username/AI-Car-Price-Prediction.git
cd AI-Car-Price-Prediction

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Train the Model (Optional)
python train.py

5️⃣ Run the Application
python app.py


Open your browser and navigate to:

http://127.0.0.1:5000/

🎮 How the Application Works

User enters car details:

Manufacturer

Kilometers Driven

Fuel Type

Car Age

Data is sent to the Flask backend via:

Form POST, or

AJAX JSON request

The ML model processes the inputs and predicts the car price

The result is displayed:

Instantly (AJAX), or

On a results page (Form submission)

📸 Application Screenshots

Add screenshots inside a /screenshots folder and reference them here.

<img src="screenshots/home.png" width="800">
<img src="screenshots/result.png" width="800">

📄 Research Publication

The research work associated with this project has been published in a peer-reviewed journal.

Title:
Machine Learning–Based Automobile Price Prediction System

Journal:
International Journal of Research and Analytical Reviews (IJRAR)

Publication Link:
https://www.ijrar.org/viewfull.php?p_id=IJRAR25D2970

This publication presents the theoretical background, dataset analysis, and model evaluation, while the current project emphasizes real-time deployment and system implementation.

🏢 Internship Context

This project was developed during my AI/ML Internship at InternPe
(Nov 24, 2025 – Dec 21, 2025).

The work focused on applying practical machine learning concepts learned during the internship, including:

Real-world data preprocessing and feature engineering

Supervised regression modeling

Backend development using Flask

Real-time ML model deployment

Robust system design with fallback mechanisms

This project represents academic and practical work completed during the internship period, emphasizing industry-relevant ML deployment practices.

📚 What I Learned

Cleaning and preprocessing real-world datasets

Building end-to-end Machine Learning pipelines

Flask backend development

AJAX-based real-time user interaction

Deploying ML models in web applications

Writing production-style project structures

🏅 Internship Acknowledgment

This project was developed as part of my InternPe Internship, focusing on:

Applied Machine Learning

Real-time prediction systems

Flask-based ML deployment

Full-stack ML application development

👤 Author

M V Karthikeya
Aspiring Machine Learning Engineer
Skills: Python • Machine Learning • Flask

📜 License

This project is licensed under the MIT License.

⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
