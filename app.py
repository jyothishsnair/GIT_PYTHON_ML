import streamlit as st
import numpy as np
import pickle
import joblib
st.title("Heart Failure Prediction Application")
heart_model = joblib.load("heart_failure_model.pkl")
heart_scaler = joblib.load("heart_failure_scaler.pkl")
age =st.slider("Age", 20, 95, 60)
anaemia = st.selectbox("Anaemia", [0, 1])
creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase", min_value=0,max_value=10000,value=500)
diabetes = st.radio("Diabetes", [0, 1],format_func=lambda x: "Yes" if x == 1 else "No")
ejection_fraction = st.number_input("Ejection Fraction", min_value=0, max_value=100, value=50)
high_blood_pressure = st.radio("High Blood Pressure", [0, 1],format_func=lambda x: "Yes" if x == 1 else "No")
platelets = st.number_input("Platelets", min_value=0, max_value=1000000, value=250000)
serum_creatinine = st.number_input("Serum Creatinine", min_value=0.0, max_value=20.0, value=1.0)
serum_sodium = st.number_input("Serum Sodium", min_value=0, max_value=200, value=135)
sex = st.selectbox("Sex (1:Male, 0:Female)",[0,1])
smoking = st.radio("Smoking", [0, 1],format_func=lambda x: "Yes" if x == 1 else "No")
time = st.number_input("Time (in days)", min_value=0, max_value=1000, value=100)

if st.button("Predict"):
    input_data = np.array([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                         high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex    , smoking, time]]) 

    scaled = heart_scaler.transform(input_data)
    prediction = heart_model.predict(scaled) 
    if prediction[0] == 1:
        st.error("The patient is at risk of heart failure.")
    else:
        st.success("The patient is not at risk of heart failure.")