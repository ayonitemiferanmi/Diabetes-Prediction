# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:07:51 2024

@author: Feranmi Ayonitemi
"""

import pandas as pd
import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler

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
def main():
    st.title("Diabetes Web App")
    
    pregnancies = st.text_input("Enter the number of pregnancies")
    glucose = st.text_input("Enter the glucose level")
    blood_pressure = st.text_input("Enter the blood pressure value")
    skin_thickness = st.text_input("Enter the skin thickness value")
    insulin = st.text_input("Enter the insulin value")
    bmi = st.text_input("Enter the bmi value")
    diabetespedigreefunction = st.text_input("Enter the diabetespedigreefunction value")
    age = st.text_input("What's the person's age")
    
    diagnosis = ""
    
    if st.button("Diabetes Predictor"):
        diagnosis = diabetes_predictor(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetespedigreefunction, age)
        
    st.success(diagnosis)

if __name__ == "__main__":
    main()