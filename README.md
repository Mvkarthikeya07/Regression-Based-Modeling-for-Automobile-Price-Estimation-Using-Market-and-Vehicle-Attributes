🚗 Regression-Based Automobile Price Estimation Using Market and Vehicle Attributes

An end-to-end Machine Learning + Web Application that predicts the market value of used cars using regression-based modeling and delivers real-time predictions through a responsive Flask web interface.

Developed during my InternPe AI/ML Internship, this project demonstrates practical skills in data preprocessing, machine learning model development, and real-time ML deployment in a web application environment.

The system combines machine learning prediction logic with a robust fallback estimation mechanism, ensuring stable and reliable results even when encountering unseen inputs.

📌 Project Overview

The automobile resale market is influenced by multiple vehicle attributes such as manufacturer, mileage, fuel type, and vehicle age. Estimating a fair resale price requires analyzing these variables collectively.

This project implements a regression-based machine learning system capable of predicting the price of a used car based on its characteristics. The trained model is integrated into a Flask-based web application, allowing users to obtain price predictions instantly through a browser interface.

The project emphasizes:

Practical machine learning system design

Clean backend architecture

Real-world data preprocessing

Production-style web deployment

🏆 Hackathon Recognition

This project was also submitted to the global hackathon:

🚀 Codegeist 2025: Atlassian Williams Racing Edition

Organized by **Atlassian, Codegeist is one of the world's largest developer hackathons focused on building innovative applications on the Atlassian ecosystem.

🎉 The project submission was among the first 300 entries worldwide, earning official Codegeist participant recognition and swag.

This participation highlights the project's:

Innovation and practical implementation

Early participation in a global developer competition

Real-world application development during a hackathon environment

📂 Hackathon Submission Proof

To maintain transparency and documentation, the official confirmation email received from Atlassian has been uploaded as a PDF.

📄 Confirmation Email (PDF):

https://drive.google.com/file/d/1bRdTqv6edJupHsh3e62eGCW5duk2dTNd/view?usp=drivesdk

This document contains:

Official confirmation from Atlassian

Proof of participation in Codegeist 2025: Atlassian Williams Racing Edition

Verification that the submission was among the first 300 entries


🚀 Key Features

🔍 Machine Learning

Cleans and preprocesses real-world used-car datasets

Handles both categorical and numerical features

Implements regression-based price prediction

Includes fallback estimation logic for unseen inputs

Easily extendable to advanced models such as:

Random Forest

Gradient Boosting

XGBoost

Neural Networks

🌐 Web Application

Flask backend with modular routing

Clean and responsive UI built with:

HTML

CSS

Bootstrap

JavaScript

Supports two prediction modes:

Form-Based Prediction

Traditional POST request submission

Displays results on a results page

AJAX Real-Time Prediction

Instant predictions

No page reload required

Smooth user experience

The application is designed with error-safe mechanisms, ensuring reliable functionality even with unexpected inputs.

🧠 Machine Learning Workflow

🧹 Data Preprocessing

The dataset undergoes multiple preprocessing steps to improve model performance and stability.

Key transformations include:

Converting kilometer strings into numerical values

Standardizing manufacturer names

Normalizing fuel-type labels

Removing extreme outliers

Handling missing values

These steps help improve model accuracy and robustness.

🔮 Prediction Logic

The system primarily relies on a trained regression machine learning model.

However, to ensure system stability under unexpected inputs, a fallback estimation formula is implemented.

predicted_price = median_price × age_factor × kms_factor

This fallback ensures that the application never fails to produce a reasonable estimate, even when the model encounters unseen combinations of attributes.

🛠 Technology Stack

Layer	Technology
Programming Language	Python
Backend Framework	Flask
Frontend	HTML, CSS, Bootstrap, JavaScript
Machine Learning	NumPy, Pandas, Scikit-learn
Data Processing	Pandas
Environment	Virtualenv
Internship	InternPe

📂 Project Structure
CAR-PRICE-PREDICTION/
│
├── app.py                # Flask backend
├── train.py              # ML training script
├── requirements.txt      # Python dependencies
│
├── data/
│   └── cars.csv          # Dataset
│
├── model/
│   └── model.pkl         # Trained ML model
│
├── static/
│   ├── css/
│   │   └── styles.css
│   │
│   └── js/
│       └── app.js
│
└── templates/
    ├── index.html
    └── result.html
    
⚙️ Installation & Setup

1️⃣ Clone the Repository
git clone https://github.com/your-username/AI-Car-Price-Prediction.git
cd AI-Car-Price-Prediction
2️⃣ Create Virtual Environment
python -m venv venv

Activate the environment:

Windows

venv\Scripts\activate

Linux / Mac

source venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Train the Model (Optional)
python train.py
5️⃣ Run the Application
python app.py

Open your browser:

http://127.0.0.1:5000/

🎮 How the Application Works

User enters vehicle details:

Manufacturer

Kilometers Driven

Fuel Type

Car Age

Data is sent to the backend through:

Form POST request

AJAX JSON request

The machine learning model predicts the car price.

The result is displayed either:

Instantly via AJAX

On a results page

📸 Application Screenshots

<img width="1366" height="768" alt="Screenshot (233)" src="https://github.com/user-attachments/assets/52b0948f-e2e4-4f34-8d78-c7d9d3a63a3d" />

<img width="1366" height="768" alt="Screenshot (234)" src="https://github.com/user-attachments/assets/244ab290-5361-48e3-ad6b-b05147c2cb31" />

📄 Research Publication

The research work associated with this project has been published in a peer-reviewed journal.

Title

Machine Learning–Based Automobile Price Prediction System

Journal

International Journal of Research and Analytical Reviews (IJRAR)

Publication Link

https://www.ijrar.org/viewfull.php?p_id=IJRAR25D2970

The research paper discusses:

Dataset analysis

Feature engineering

Model evaluation

The current repository focuses on practical system implementation and deployment.

🏢 Internship Context

This project was developed during my AI/ML Internship at InternPe (Nov 24, 2025 – Dec 21, 2025).

The internship emphasized practical machine learning skills such as:

Real-world dataset preprocessing

Supervised regression modeling

Flask-based ML deployment

Full-stack ML application development

Production-style system design

📚 Key Learnings

Through this project I gained practical experience in:

Cleaning and preprocessing real-world datasets

Designing machine learning pipelines

Building Flask-based backend systems

Implementing AJAX real-time predictions

Deploying ML models in web applications

Structuring production-ready ML projects

👤 Author

M V Karthikeya
Aspiring Machine Learning Engineer

Skills:

Python

Machine Learning

Flask

Data Processing

Web-based ML Deployment

📜 License

This project is licensed under the MIT License.

⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
