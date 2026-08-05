📚 EduPro – Predictive Modeling for Course Demand & Revenue Forecasting
📖 Overview

EduPro is an end-to-end Machine Learning and Data Analytics project developed to help online learning platforms make data-driven decisions through predictive intelligence. Instead of relying solely on historical reports, the application forecasts future course enrollments, course revenue, and category-level revenue, enabling proactive planning for course launches, pricing strategies, instructor allocation, and business growth.

The project combines data preprocessing, feature engineering, predictive modeling, model evaluation, and an interactive Streamlit dashboard to provide real-time insights for educational stakeholders.

🎯 Problem Statement

Online learning platforms often rely on historical intuition when making decisions regarding:

Launching new courses
Adjusting course pricing
Onboarding instructors
Allocating resources
Identifying high-demand course categories

EduPro addresses this challenge by leveraging machine learning to accurately predict future demand and revenue, enabling informed business decisions.

🚀 Features
📊 Interactive Exploratory Data Analysis (EDA)
📈 Course Enrollment Forecasting
💰 Course Revenue Prediction
📚 Category Revenue Forecasting
🤖 Multiple Machine Learning Models
📉 Model Performance Comparison
🔍 Feature Importance Analysis
💡 Business Insights & Recommendations
📂 Dataset Explorer
⚡ Live Interactive Analytics
🎨 Professional Multi-Page Streamlit Dashboard
🎯 Predictive Targets
Enrollment Count
Course Revenue
Category Revenue
📂 Dataset

The project utilizes four interconnected datasets containing information related to:

Courses
Teachers
Transactions
Supporting educational data

The data is merged, cleaned, and transformed before model development.

🛠 Feature Engineering

Engineered features include:

Price Bands
Duration Buckets
Rating Tiers
Course Level Encoding
Teacher Experience Buckets
Teacher Rating Score
Expertise–Category Match Score
Historical Enrollment Count
Historical Average Revenue
Revenue per Enrollment
🤖 Machine Learning Models

The following regression models are implemented and compared:

Linear Regression
Ridge Regression
Lasso Regression
Random Forest Regressor
Gradient Boosting Regressor

The application automatically selects the best-performing model based on evaluation metrics.

📏 Model Evaluation

Models are evaluated using:

Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
R² Score
📊 Dashboard Modules
🏠 Dashboard
Project Overview
KPI Cards
Revenue & Demand Summary
📈 Exploratory Data Analysis
Dataset Overview
Distribution Analysis
Correlation Heatmaps
Category Insights
Instructor Insights
🤖 Model Performance
Model Comparison
Evaluation Metrics
Best Model Selection
🎯 Live Prediction Studio
User Input Form
Enrollment Prediction
Course Revenue Prediction
Category Revenue Prediction
🔍 Feature Importance
Feature Importance Charts
Demand Driver Analysis
Model Explainability
💡 Business Insights
Pricing Recommendations
Course Launch Suggestions
Strategic Planning Insights
📂 Dataset Explorer
Merged Dataset Preview
Dataset Statistics
Feature Information
🖥 Tech Stack
Programming Language
Python
Data Analysis
Pandas
NumPy
Machine Learning
Scikit-learn
Data Visualization
Plotly
Matplotlib
Seaborn
Web Framework
Streamlit
📁 Project Structure
EduPro_Predictive_Analytics/
│
├── app.py
├── modeling.py
├── data_pipeline.py
├── requirements.txt
├── README.md
├── notebooks/
├── data/
├── .streamlit/
└── assets/
⚙️ Installation

Clone the repository:

git clone https://github.com/your-username/EduPro_Predictive_Analytics.git

Navigate to the project directory:

cd EduPro_Predictive_Analytics

Create a virtual environment:

python -m venv venv

Activate the virtual environment:

Windows

venv\Scripts\activate

Linux / macOS

source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py
📸 Dashboard Highlights
Interactive Multi-Page Dashboard
Live Machine Learning Predictions
Real-Time Business Intelligence
Feature Importance Visualization
Professional Enterprise UI
Responsive Design
Interactive Charts & Tables
📌 Future Enhancements
Deep Learning-based forecasting
Time-series demand prediction
Automated model retraining
Cloud deployment
User authentication
AI-generated business reports
API integration for real-time data
🎓 Project Objective

EduPro demonstrates how predictive analytics can transform educational planning by forecasting future demand and revenue. The application empowers institutions to optimize pricing, launch high-potential courses, improve instructor allocation, and make evidence-based strategic decisions through an intuitive analytics dashboard.

📄 License

This project is intended for educational and academic purposes. Feel free to fork, learn from, and extend it for research or portfolio use.

Author

Chandni Kumar 
Linkedin - https://www.linkedin.com/in/chandni-k25/
