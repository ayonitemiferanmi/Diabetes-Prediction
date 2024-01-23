# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


with open("C:/Users/HP/Desktop/Summer Program/Machine Learning/Classification/Diabetes Prediction/diabetes_predictor.pkl", "rb") as f:
    loaded_model = pickle.load(f)
    
def diabetes_predictor(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetespedigreefunction, age):
    # The dictionary of the predictor
    predictor_dictionary = {
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetespedigreefunction],
        "Age": [age]
    }
    
    # Convert the dictionary to a dataframe
    predictor_data = pd.DataFrame(predictor_dictionary)
    
    # Data Standardization
    #scaler = StandardScaler()
    #predictor_data_scaled = scaler.fit_transform(predictor_data)
    
    
    
    # Make predicitons
    predictor = loaded_model.predict(predictor_data)
    
    
    # Conditional statement
    if predictor == 0:
        return "The patient is Non-diabetic"
    else:
        return "The patient is diabetic"

diabetes_predictor(6, 148, 72, 35, 0, 33.6, 0.627, 50)
