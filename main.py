import streamlit as st
import pandas as pd
import pickle

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.header("AI Diabetes Estimator")
with st.form('input_form'):
    st.subheader("Fill in the details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", value=20, min_value=0)
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
    
    with col2:
        smoking_history = st.selectbox("Smoking History", ["Never", "Former", "Current"])
        blood_glucose_level = st.number_input("Blood Glucose Level (mg/dL)", min_value=0.0, value=100.0)
        hba1c_level = st.number_input("HbA1c(glycated hemoglobin) Level (%)", min_value=0.0, value=5.5)
        bmi = st.number_input("BMI", min_value=0.0, value=25.0,)
    
    submit_button = st.form_submit_button(label='Submit')
    smoking_status = [0, 0, 0]
    if submit_button:
        smoking_status = [0, 0, 0]
        if smoking_history.lower() == "never":
            smoking_status[0] = 1
        elif smoking_history.lower() == "former":
            smoking_status[1] = 1
        elif smoking_history.lower() == "current":
            smoking_status[2] = 1
    
    input_data = pd.DataFrame([{
            "gender": 0 if gender.lower() == "male" else 1,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "bmi": bmi,
            "HbA1c_level": hba1c_level,
            "blood_glucose_level": blood_glucose_level,
            "smoking_history_current": smoking_status[2],
            "smoking_history_former": smoking_status[1],
            "smoking_history_never": smoking_status[0],
        }])
    
    if submit_button:
        input_scaled = scaler.transform(input_data)
        input_scaled = pd.DataFrame(input_scaled, columns=input_data.columns)
        prediction = model.predict(input_scaled)[0]
        result_class = "Success" if prediction == 0 else "Error"
        
        if prediction == 0:
            st.success("Congrats! You don't have diabetes")
        else:
            st.error("You have been diagnosed with diabetes")

    
