Overview
This project is a machine learning-based web application that predicts multiple diseases, including:
✔ Diabetes
✔ Heart Disease
✔ Parkinson's Disease
✔ Lung Cancer
✔ Hypo-Thyroid Disease

The application is built using Streamlit and leverages trained machine learning models to provide predictions based on user inputs.

Features
✅ Multi-Disease Prediction: Supports predictions for multiple diseases
✅ Interactive User Interface: Built with Streamlit
✅ Pre-Trained Machine Learning Models: Ensures quick and accurate results
✅ Dataset Integration: Uses preprocessed datasets for better efficiency
✅ Instant Diagnosis: Real-time results based on input parameters

Technologies Used
Python
Streamlit (for the web interface)
Scikit-Learn (for machine learning)
Pandas & NumPy (for data processing)
Pickle (for model storage and loading)
Installation Guide
1. Clone the Repository
https://github.com/tushant687/Implementation-of-AI-Powered-Medical-Diagnosis-.git
2. Install Dependencies
pip install -r requirements.txt
3. Run the Application
streamlit run app.py
Project Structure
disease-prediction/
│-- Models/                        # Pre-trained machine learning models  
│   ├── diabetes_model.sav  
│   ├── heart_disease_model.sav  
│   ├── parkinsons_model.sav  
│   ├── lungs_disease_model.sav  
│   ├── thyroid_model.sav  
│-- Data/                          # Processed and raw datasets  
│   ├── diabetes_data.csv  
│   ├── hypothyroid.csv  
│   ├── parkinson_data.csv  
│   ├── prepocessed_hypothyroid.csv  
│   ├── prepocessed_lungs_data.csv  
│   ├── survey_lung_cancer.csv  
│-- app.py                         # Main Streamlit web application  
│-- requirements.txt                # Required dependencies  
│-- README.md                       # Project documentation  
How It Works
The user selects a disease from the dropdown menu.
They input relevant health parameters (e.g., age, glucose levels, heart rate).
The app uses a trained ML model to make a prediction.
The result is displayed, indicating whether the user is at risk or not.
Datasets Used
The models are trained on various medical datasets:

Diabetes Dataset (diabetes_data.csv)
Heart Disease Dataset
Parkinson's Disease Dataset (parkinson_data.csv)
Lung Cancer Survey Dataset (survey_lung_cancer.csv)
Hypothyroid Dataset (hypothyroid.csv)
Future Enhancements
🚀 Improve Model Accuracy with deep learning techniques
🚀 Add More Disease Predictions such as Alzheimer's and Stroke
🚀 Integrate a User Dashboard for tracking past results
🚀 Deploy the App Online using Heroku or AWS

License
This project is licensed under the MIT License.
