import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
# Assuming exampleModel.pkl is in the same directory

st.set_page_config(page_title="Health Condition Predictor", layout="centered")

st.title('Health Condition Predictor')
st.markdown("### Enter your health statistics to get a prediction")

# Input fields for all features (excluding 'id')
with st.sidebar:
    st.header("User Input Features")

    # Numerical features - using number_input or sliders
    sleep_duration = st.number_input('Sleep Duration (hours)', min_value=0.0, max_value=24.0, value=7.0, step=0.1)
    heart_rate = st.number_input('Heart Rate (bpm)', min_value=30, max_value=200, value=70)
    bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    calorie_expenditure = st.number_input('Calorie Expenditure (kcal)', min_value=0.0, max_value=5000.0, value=2000.0,
                                          step=10.0)
    steps = st.number_input('Steps', min_value=0, max_value=30000, value=7000)
    blood_pressure_systolic = st.number_input('Blood Pressure Systolic (mmHg)', min_value=50, max_value=250, value=120)
    blood_pressure_diastolic = st.number_input('Blood Pressure Diastolic (mmHg)', min_value=30, max_value=150, value=80)
    oldpeak = st.number_input('Oldpeak (ST depression induced by exercise relative to rest)', min_value=0.0,
                              max_value=6.0, value=1.0, step=0.1)
    cholesterol = st.number_input('Cholesterol (mg/dL)', min_value=100, max_value=400, value=200)
    glucose = st.number_input('Glucose (mg/dL)', min_value=50, max_value=300, value=100)
    age = st.number_input('Age (years)', min_value=0, max_value=120, value=30)
    stress_level = st.slider('Stress Level (1-5)', min_value=1, max_value=5, value=3)

    # Categorical features
    gender = st.selectbox('Gender', ['male', 'female', 'other'])

# Create a DataFrame from the inputs
input_data = pd.DataFrame([{
    'id': 0,  # Placeholder id, not used for prediction but expected by preprocessor
    'sleep_duration': sleep_duration,
    'heart_rate': heart_rate,
    'bmi': bmi,
    'calorie_expenditure': calorie_expenditure,
    'steps': steps,
    'blood_pressure_systolic': blood_pressure_systolic,
    'blood_pressure_diastolic': blood_pressure_diastolic,
    'oldpeak': oldpeak,
    'cholesterol': cholesterol,
    'glucose': glucose,
    'gender': gender,
    'age': age,
    'stress_level': stress_level
}])

# Make prediction
if st.button('Predict Health Condition'):
    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        st.subheader('Prediction Result:')
        st.success(f"The predicted health condition is: **{prediction[0]}**")

        st.subheader('Prediction Probabilities:')
        proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
        st.dataframe(proba_df.T.rename(columns={0: 'Probability'}))

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
