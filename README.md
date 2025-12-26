🚗 AI-Powered Car Price Prediction Web App
Real-Time Machine Learning–Based Used Car Valuation (Flask + ML)
<p align="center"> <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python"> <img src="https://img.shields.io/badge/Flask-Web%20Framework-black?style=for-the-badge&logo=flask"> <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge&logo=scikitlearn"> <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge"> </p>
📌 Overview

This project is a complete end-to-end Machine Learning + Web Application, developed during my InternPe Internship.

The application predicts the market price of a used car based on key vehicle attributes using a regression-based ML model and provides real-time predictions through a responsive Flask web interface.

The project demonstrates real-world ML deployment, clean backend architecture, and production-style design principles.

🚀 Key Features
🔍 Machine Learning

Cleans and preprocesses real-world used-car data

Handles numerical and categorical features

Regression-based price prediction

Fallback median-based estimation to ensure stability

Easily extendable to Random Forest, XGBoost, etc.

🌐 Web Application

Flask backend with organized routing

Clean, responsive UI using HTML, CSS, Bootstrap

Supports:

Traditional Form POST predictions

AJAX-based real-time predictions (no page reload)

Error-safe design for a smooth user experience

🧠 Machine Learning Workflow
🧹 Data Preprocessing

Converts kilometer strings → numeric values

Standardizes manufacturer names

Normalizes fuel type labels

Removes outliers for improved prediction stability

🔮 Prediction Logic

ML regression model predicts car price

If unseen or unstable inputs occur, a fallback formula ensures reliability:

predicted_price = median_price × age_factor × kms_factor


This guarantees robust predictions without system failure.

🛠 Tech Stack
Layer	Technology
Language	Python
Backend	Flask
Frontend	HTML, CSS, Bootstrap, JavaScript
Machine Learning	NumPy, Pandas, Scikit-learn
Environment	Virtualenv
Internship	InternPe
📂 Project Structure
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

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/AI-Car-Price-Prediction.git
cd AI-Car-Price-Prediction

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Train the Model (Optional)
python train.py

5️⃣ Run the Application
python app.py


Open your browser:

http://127.0.0.1:5000/

🎮 How the App Works

User enters car details:

Manufacturer

Kilometers Driven

Fuel Type

Car Age

Data is sent to the Flask backend via:

Form POST, or

AJAX JSON request

ML model processes the inputs and predicts the price

Result is displayed:

Instantly (AJAX), or

On a results page (Form submission)

📸 Application Screenshots
<img width="1366" height="768" src="https://github.com/user-attachments/assets/fbfc4a87-7713-4caa-a8da-9cd54c172603" />
<img width="1366" height="768" src="https://github.com/user-attachments/assets/f9344be3-b20a-43fb-b661-52ed2a425799" />
<img width="1366" height="768" src="https://github.com/user-attachments/assets/c49cc610-9e4e-4462-af64-5d657e69c51c" />

📚 What I Learned

Cleaning and preprocessing real-world datasets

Building end-to-end ML pipelines

Flask backend development

AJAX-based real-time user interaction

Deploying ML models in web applications

Writing production-style project structure

🏅 Internship Acknowledgment

This project was developed as part of my InternPe Internship, focusing on:

Applied Machine Learning

Real-time prediction systems

Flask-based ML deployment

Full-stack ML application development

👤 Author

M V Karthikeya
Aspiring Machine Learning Engineer
Python | Machine Learning | Flask

⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
