import streamlit as st
import numpy as np
import joblib

# 🎯 App Title
st.title("Heart Failure Prediction Application")

# 🖼️ Add background image using CSS
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://copilot.microsoft.com/th/id/BCO.c4134932-11dd-4972-91a1-aecf0ba5c2eb.png");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.7);
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# 🖌️ Custom text color styling (yellow or white)
text_color_css = """
<style>
h1, h2, h3, h4, h5, h6, p, label, div {
    color: yellow !important;  /* change to white if you prefer */
}
</style>
"""
st.markdown(text_color_css, unsafe_allow_html=True)

# 🧠 Load model and scaler
heart_model = joblib.load("heart_failure_model.pkl")
heart_scaler = joblib.load("heart_failure_scaler.pkl")

# 🩺 Input layout
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=60)
    anaemia = st.selectbox("Anaemia", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    high_bp = st.selectbox("High Blood Pressure", ["No", "Yes"])
    sex = st.selectbox("Sex", ["Female", "Male"])
with col2:
    cpk = st.number_input("Creatinine Phosphokinase", min_value=0, max_value=10000, value=300)
    ef = st.number_input("Ejection Fraction", min_value=10, max_value=80, value=35)
    platelets = st.number_input("Platelets", min_value=0, max_value=1000000, value=250000)
    sc = st.number_input("Serum Creatinine", min_value=0.0, max_value=20.0, value=1.0)
    ss = st.number_input("Serum Sodium", min_value=0, max_value=200, value=135)
time = st.number_input("Follow-up Time (days)", min_value=0, max_value=1000, value=100)
smoke = st.selectbox("Smoking Status", ["No", "Yes"])

# 🔢 Encode categorical inputs
anaemia_val = 1 if anaemia == "Yes" else 0
diabetes_val = 1 if diabetes == "Yes" else 0
high_bp_val = 1 if high_bp == "Yes" else 0
sex_val = 1 if sex == "Male" else 0
smoke_val = 1 if smoke == "Yes" else 0
# 🚀 Prediction
if st.button("Predict"):
    input_data = np.array([[age, anaemia_val, cpk, diabetes_val, ef,
                            high_bp_val, platelets, sc, ss, sex_val, time, smoke_val]])
    scaled = heart_scaler.transform(input_data)
    prediction = heart_model.predict(scaled)

    if prediction[0] == 1:
        st.markdown(
            f"<div style='background-color:#660000;padding:15px;border-radius:10px;'>"
            f"<h3>⚠️ High Risk of Heart Failure</h3>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='background-color:#004d00;padding:15px;border-radius:10px;'>"
            f"<h3>✅ Low Risk of Heart Failure</h3>"
            f"</div>",
            unsafe_allow_html=True
        )

# ✍️ Footer credit
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        color: yellow; /* change to white if you prefer */
        font-size: 16px;
        padding: 10px;
    }
    </style>
    <div class="footer">
        Created by Jyothish
    </div>
    """,
    unsafe_allow_html=True
)
